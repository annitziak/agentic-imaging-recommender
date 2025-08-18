import requests
import xml.etree.ElementTree as ET
import os
import time
import Post_Filtering_Agent
import pickle
import re
import json

class Medical_Review_Agent:
    def __init__(self, api_key=None, batch_size=100, delay=0.11):
        self.api_key = api_key or os.getenv("PUBMED_API_KEY")
        self.batch_size = batch_size # Number of PMIDs to fetch in one request to avoid rate limits
        self.delay = delay # Delay between requests to avoid hitting API limits
        self.generated_abstracts = {}
        self.post_filter = Post_Filtering_Agent.PostFilteringAgent()

    def make_prompts_for_deepretrieval(self, condition, procedure=None):
        """ Generates a list of prompts for DeepRetrieval based on the condition and optional procedure.
        You will need to deploy the DeepRetrieval model separately and use these prompts to query it. 
        You can sh into their vllm_host.sh, for more info github.com/pat-jj/DeepRetrieval"""
        prompts = [
            f"{condition}",
            f"Diagnostic imaging for {condition}",
            f"Clinical evidence for the use of diagnostic imaging in the evaluation of {condition}",
            f"{condition} and related conditions",
            f"Affected conditions: {condition} and synonyms",
            f"Alternative terminology for {condition}",
        ]
        if procedure:
            prompts.extend([
                f"Clinical evidence for the use of {procedure} in the evaluation of {condition}",
                f"P: {condition}; I: {procedure}; C: Alternative imaging procedures; O: Diagnostic accuracy, risk, and benefits"
            ])
        return prompts

    def search_pubmed(self, query, max_results=20):
        """ Searches PubMed using the provided query and returns a list of PMIDs.
        Filters results to include only English abstracts from 2013 onwards."""

        filtered_query = (
            f"({query}) AND (\"2013/01/01\"[PDAT] : \"3000\"[PDAT]) "
            "AND (english[lang])"
        )
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': filtered_query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        # This will make extraction faster
        if self.api_key:
            params['api_key'] = self.api_key

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()['esearchresult']['idlist'] #get the PMIDS list

    def fetch_pubmed_details(self, pmids):
        """ Fetches detailed information for a list of PMIDs from PubMed."""

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi" # the URL for fetching detailed records
        results = []
        for i in range(0, len(pmids), self.batch_size): # batch processing
            batch_pmids = pmids[i:i + self.batch_size]
            print(f" Fetching batch {i // self.batch_size + 1}/{(len(pmids) - 1) // self.batch_size + 1}")
            params = {
                "db": "pubmed",
                "id": ",".join(batch_pmids),
                "retmode": "xml"
            }
            if self.api_key:
                params["api_key"] = self.api_key

            resp = requests.get(url, params=params)
            resp.raise_for_status()

            # Parse the XML response: get the relevant fields needed for the agent
            root = ET.fromstring(resp.content)
            for article in root.findall('.//PubmedArticle'):
                pmid = article.findtext('.//PMID')
                title = article.findtext('.//ArticleTitle')
                journal = article.findtext('.//Journal/Title')
                year = article.findtext('.//PubDate/Year')
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ""
                pub_types = [pt.text for pt in article.findall('.//PublicationType') if pt.text]

                # The information we are retrieving for each abstract 
                results.append({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "abstract": abstract.strip() if abstract else "",
                    "publication_types": pub_types
                })

            time.sleep(self.delay) # be polite :)
        return results

    # This assumes that the DeepRetrieval queries are already generated, using the queries from make_prompts_for_deepretrieval
    # They should be in json format with a "condition" key and "rewritten_queries". 
    # The added term "Diagnostic Imaging" has been added and the queries have been cleaned and duplicate removal was done
    # See README for details.
    # An example can be found in data/deepretrieval_queries_generalization_set_example.json

    def process_variants(self, queries_deepretrieval):
        """ Processes a list of variant queries, fetching abstracts from PubMed for each condition."""
        for variant in queries_deepretrieval:
            condition = variant["condition"]
            queries = variant.get("rewritten_queries") 
            pmids = set()
            for _, q in enumerate(queries, 1):
                ids = self.search_pubmed(q, max_results=25) #get the top k results for each query
                pmids.update(ids)

            print(f"Total unique PMIDs for {condition}: {len(pmids)}")
            # Fetch details for the PMIDs
            results = self.fetch_pubmed_details(list(pmids))

            # If abstracts are empty, we skip them (this is a current limitation)
            filtered = [
                r for r in results
                if r["abstract"] and "abstract not available" not in r["abstract"].lower()
            ]

            self.generated_abstracts[condition] = filtered
        return self.generated_abstracts

    def predict_strength_of_evidence(self, generated_abstracts):
        """
        Adds 'predicted_quality' to each paper per condition using the PostFilteringAgent.
        """
        results = {}
        for condition, papers in generated_abstracts.items():
            soe_prediction = self.post_filter.predict_quality(papers) # This uses the PostFilteringAgent to predict quality
            results[condition] = soe_prediction # assign the predicted quality to the condition
        return results
    
    def filter_high_quality_abstracts(self, predicted_abstracts_by_condition, min_strength=3):
        """ Filters abstracts based on the predicted strength of evidence."""
        filtered_results = {}

        # For each condition, filter papers based on the predicted quality
        for condition, entries in predicted_abstracts_by_condition.items():
            filtered_papers = []
            for entry in entries:
                paper = entry["paper"]
                strength = entry["predicted_quality"]
                # If the strength is above the minimum, we keep the paper
                if strength >= min_strength: 
                    paper["predicted_strength"] = int(strength)
                    filtered_papers.append(paper)
            filtered_results[condition] = filtered_papers
            print(f"Condition: {condition}, Filtered {len(filtered_papers)} papers with strength >= {min_strength}")

        return filtered_results

    def condense_abstracts(self, abstracts):
        """
        Condenses the high-quality abstracts in each citation to key sections and filters entries.
        whose total abstract word count exceeds the limit.
        """
        # Define section labels and regex for matching (these are common in Pubmed abstracts)
        section_labels = [
            "materials and methods", "material and methods", "patients and methods", "methods and materials",
            "background", "purpose", "objective", "objectives", "aim", "aims", "introduction", "methods", "materials",
            "patients", "subjects", "design",
            "results", "findings", "main outcome", "main results", "outcomes", "key results", "observations",
            "conclusion", "conclusions", "interpretation", "summary", "key message", "main message"
        ]
        keep_labels = {
            "results", "findings", "main outcome", "main results", "outcomes", "key results", "observations",
            "conclusion", "conclusions", "interpretation", "summary", "key message", "main message"
        }
        section_regex = re.compile(
            r"(" + "|".join(re.escape(lbl.upper()) for lbl in section_labels) + r")\s*(:|&|AND|\n{1,2})"
        )

        def extract_sections(abstract):
            """Extracts key sections from the abstract based on predefined labels."""
            upper = abstract.upper() #sometimes they are capitalized differently too
            matches = list(section_regex.finditer(upper))
            if not matches:
                return abstract.strip() # If no sections are found, return the original abstract because we do not know the main conclusions

            spans = [m.start() for m in matches] + [len(abstract)]
            sections = [
                (matches[i].group(1).lower(), abstract[spans[i]:spans[i+1]].strip())
                for i in range(len(matches))
            ]

            selected = []
            if matches[0].start() > 0:
                selected.append(abstract[:matches[0].start()].strip())
            else:
                selected.append(sections[0][1])

            for label, text in sections:
                if label in keep_labels and text not in selected:
                    selected.append(text)

            # return the condensed sections, ensuring no empty strings
            return "\n\n".join([s for s in selected if s])

        for _, papers in abstracts.items():
            for paper in papers:
                abstract = paper.get("abstract", "")
                paper["abstract"] = extract_sections(abstract)

        return abstracts
    
    def redirect_in_variants(self, generated_abstracts, dataset):
        """
        Redirects the generated abstracts to the PostFilteringAgent for further processing.
        This is a placeholder for any additional processing that might be needed.
        """
        # Load the ontology data, this can also be extended to include UMLS terms, for now simplified for the conditions in our generalization set
        with open("data/ontology_examples.json", "r") as f:
            ontology = json.load(f)

        PROCEDURE_SYNONYMS = ontology["procedure_synonyms"]
        CONDITION_SYNONYMS = ontology["condition_synonyms"]
        EVALUATION_KEYWORDS = ontology["evaluation_keywords"]
        INCLUDED_PUB_TYPES = set(pt.lower() for pt in ontology["included_pub_types"])
        PROCEDURE_CORES = ontology["procedures_cores"]

        print(f"Using {len(PROCEDURE_SYNONYMS)} procedure synonyms and {len(CONDITION_SYNONYMS)} condition synonyms")

        def contains_any(text, keywords):
            """Checks if the text contains any of the keywords."""
            for kw in keywords:
                if " " in kw:
                    if kw in text.lower():
                        return True
                else:
                    pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(pattern, text.lower()):
                        return True
            return False
        
        
        def strip_procedure(proc):
            """ Cleans the procedure string to find the core procedure name. These are common in Pubmed abstracts.
            For example "CT scan of the abdomen with IV Contrast" -> "CT scan"
            """
            proc_lower = proc.lower()
            proc_lower = re.sub(r'\(.*?\)', '', proc_lower)  # remove parentheses
            proc_lower = re.sub(r'\s+', ' ', proc_lower).strip()  # clean spaces

            # Try to find LONGEST matching core (clean up excess words)
            for core in sorted(PROCEDURE_CORES, key=len, reverse=True):
                if core in proc_lower:
                    return core
            return proc_lower  # fallback 
                
        def is_relevant_paper(paper, proc_syns, cond_syns):
            """ 
            Checks if the paper is relevant based on procedure and condition synonyms, evaluation keywords, and publication types.
            Using the heuristic rule: the abstract must contain both a procedure and a condition, and either an evaluation keyword or an 
            included publication type to be included in condition-variant-procedure triplet.
            """
            abstract = paper.get('abstract', '').lower()
            pub_types = [pt.lower() for pt in paper.get('publication_types', [])]

            proc_pattern = r'\b(' + '|'.join(re.escape(s) for s in proc_syns) + r')\b'
            cond_pattern = r'\b(' + '|'.join(re.escape(s) for s in cond_syns) + r')\b'
            
            procedure_mentioned = re.search(proc_pattern, abstract) # Check if any procedure synonym is mentioned
            condition_mentioned = re.search(cond_pattern, abstract) # Check if any condition synonym is mentioned
            evaluation_mentioned = contains_any(abstract, EVALUATION_KEYWORDS) # Check if any evaluation keyword is mentioned
            pub_type_match = any(pt in INCLUDED_PUB_TYPES for pt in pub_types) # Check if any included publication type is mentioned

            return procedure_mentioned and condition_mentioned and (evaluation_mentioned or pub_type_match)
        
        # Initialize the processed list to hold the final results
        processed = []

        # Iterate through the dataset entries, which should contain conditions, variants and procedures
        for entry in dataset:
            condition = entry.get("condition", "").strip()
            procedure = entry.get("procedure", "")
            procedure_core = strip_procedure(procedure) #strip to the core procedure name
            print(f"Processing condition: {condition}, procedure: {procedure_core}")

            # Get synonyms for matching
            proc_syns = PROCEDURE_SYNONYMS.get(procedure_core, [procedure_core])
            cond_syns = CONDITION_SYNONYMS.get(condition.lower(), [condition.lower()])
            
            # Filter abstracts for condition and procedure
            abstracts_for_condition = generated_abstracts.get(condition, [])
            if not abstracts_for_condition:
                print(f"No abstracts found for condition: {condition}")
                continue

            print(f"Processing {len(abstracts_for_condition)} abstracts for condition: {condition}")
            filtered_papers = []
            for paper in abstracts_for_condition:
                # for each paper find if its relevant based on the procedure and condition synonyms
                if is_relevant_paper(paper, proc_syns, cond_syns):
                    filtered_papers.append(paper)
            print(f"Found {len(filtered_papers)} relevant papers")

            # Build new citations dict: "1" -> abstract text, "2" -> abstract text, ...
            citations = {}
            for i, p in enumerate(filtered_papers, start=1):
                abs_text = p.get("abstract", "").strip()
                if abs_text:
                    citations[str(i)] = abs_text

            # Construct new entry with original procedure name and updated citations
            new_entry = entry.copy()
            new_entry["citations"] = citations

            processed.append(new_entry)

        return processed
    def process_generated_abstracts_per_condition(self, generated_abstracts, generalization_dataset=None):
        """
        Full pipeline for retrieval: predicts strength of evidence, then filters high quality abstracts per condition.
        After that it condenses them to the results/conclusions sections and redirects them to condition-variant-procedure triplets.
        """
        predicted = self.predict_strength_of_evidence(generated_abstracts)
        print(f"Predicted strength of evidence")
        filtered = self.filter_high_quality_abstracts(predicted)
        print(f"Filtered high quality abstracts")
        condensed = self.condense_abstracts(filtered)
        print(f"Condensed abstracts to main sections")
        # Redirect to PostFilteringAgent for further processing
        redirected = self.redirect_in_variants(condensed, dataset=generalization_dataset)
        print(f"Redirected abstracts in variants")

        return redirected

if __name__ == "__main__":
    # Load your variant queries from a JSON file (replace path as needed)
    with open("data/deepretrieval_queries_generalization_set.json", "r") as f:
        variant_queries = json.load(f)

    # Initialize the agent (optionally pass your PubMed API key here)
    agent = Medical_Review_Agent(api_key="YOUR_PUBMED_API_KEY")

    # Step 1: Search and fetch abstracts from PubMed
    print("Starting PubMed scraping...")
    generated_abstracts = agent.process_variants(variant_queries)

    # Open the generalization dataset to redirect the abstracts (this contains the condition-variant-procedure triplets and we will append the citations)
    input_path = ""
    with open(input_path, "r") as f:
        generalization_dataset = json.load(f)

    # Step 2: Predict evidence strength, filter, condense, and redirect
    print(" Processing abstracts through full pipeline...")
    processed_abstracts = agent.process_generated_abstracts_per_condition(generated_abstracts, generalization_dataset)

    # Step 3: Save results
    output_path = ""
    with open(output_path, "w") as f:
        json.dump(processed_abstracts, f, indent=2)

    print(f"✅ Done! Saved {len(processed_abstracts)} processed entries to {output_path}")