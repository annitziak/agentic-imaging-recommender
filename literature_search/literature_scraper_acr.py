import os
import re
import json
import pdfplumber
import requests
import xml.etree.ElementTree as ET
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Parses ACR Appendix documents to extract all relevant evidence.
# For each condition–variant–procedure pair, it:
#   1. Reads and processes the PDFs.
#   2. Identifies the required citations.
#   3. Retrieves corresponding study information from PubMed.

API_KEY = os.getenv("NCBI_KEY") or "your_ncbi_api_key_here" # Replace with your actual NCBI API key

def fetch_pubmed_abstract(pmid, api_key=API_KEY):
    """Fetch PubMed abstract for a given PMID and using the provided API key. The rate is limited to 3 requests per second."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    params = {
        "db": "pubmed",
        "id": pmid,
        "rettype": "abstract",
        "retmode": "xml",
        "api_key": api_key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        abstract_texts = [
            f"{elem.attrib.get('Label')}: {elem.text.strip()}" if elem.attrib.get('Label') else elem.text.strip()
            for elem in root.findall(".//Abstract/AbstractText") if elem.text
        ]
        return " ".join(abstract_texts).strip()
    except Exception as e:
        print(f"Warning: Failed to fetch PMID {pmid}: {e}")
        return None

# These helper functions help with the parsing of the PDF files
def is_variant_line(line):
    """Check if a line indicates a variant, e.g., 'Variant X: ...'."""
    return line.strip().startswith("Variant ") and ":" in line

def is_reference_line(line):
    """Check if a line is a reference line, e.g., '1 (12345678)' or '2 (12345678)'. This is a simplified check."""
    return re.match(r"^\d+\s*\(\d{7,8}\)", line.strip())


def parse_pdf_variants(pdf_path):
    """Parse the PDF to extract variants and their associated citations. They are in the form <Variant X: ...>"""
    with pdfplumber.open(pdf_path) as pdf:
        lines = [line for page in pdf.pages for line in (page.extract_text() or "").split("\n")]

    results = []
    variant, i = None, 0
    while i < len(lines):
        line = lines[i].strip()
        if is_variant_line(line):
            variant = line
            i += 1
            continue
        
        # Given the structure of the ACR PDFs
        if "References Study Quality" in line:
            citations = {}
            i += 1
            while i < len(lines):
                ref_line = lines[i].strip()
                if not ref_line or not (is_reference_line(ref_line) or "References Study Quality" in ref_line):
                    break
                for match in re.finditer(r"(\d+)\s*\((\d{7,8})\)", ref_line):
                    ref_num, pmid = match.groups()
                    citations[ref_num] = pmid  # store PMID only
                i += 1
            if variant and citations:
                results.append({
                    "variant": variant,
                    "procedure": "",
                    "citations": citations
                })
            continue
        i += 1
    return results

def process_pdfs():
    """Process all PDFs in the input directory, parse variants and citations, and save results."""

    # Note : This script does not handle the case where a procedure does not have any citations. These need to be manually added.
    # It also does not capture the procedure name, due to multiline issues so it need to be manually added. 
    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith("_Apdx.pdf"):
            continue

        pdf_path = os.path.join(INPUT_DIR, fname)
        print(f"Parsing {fname}")
        parsed_variants = parse_pdf_variants(pdf_path)

        for entry in parsed_variants:
            for ref_num, pmid in entry["citations"].items():
                abstract = fetch_pubmed_abstract(pmid)
                entry["citations"][ref_num] = {
                    "pmid": pmid,
                    "abstract": abstract
                }

        clinical_condition = fname.replace("_Apdx.pdf", "")
        output_path = os.path.join(OUTPUT_DIR, f"{clinical_condition}_acr_citations_variant_specific.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(parsed_variants, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_path}")

# If you want to merge all outputs into a single file, you can use this function and give it the directory where all the JSON files are stored.
def merge_all_outputs(output_path):
    """Merge all JSON files in the OUTPUT_DIR into a single JSON file."""
    merged = []
    for fname in os.listdir(OUTPUT_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(OUTPUT_DIR, fname), "r", encoding="utf-8") as f:
            print(f" Merging {fname}")
            data = json.load(f)
            merged.extend(data if isinstance(data, list) else [data])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(merged)} entries into {output_path}")
    return merged

def analyze_citation_lengths(entries):
    """Simple analysis the total word count of abstracts in citations for each entry and plots the distribution.
    This is to find what the appropriate context window for our LLM should be."""
    entry_totals = []
    for entry in entries:
        citations = entry.get("citations", {})
        total_words = sum(
            len(str(c.get("abstract", "")).split()) 
            for c in citations.values() if isinstance(c, dict) and c.get("abstract")
        )
        entry_totals.append(total_words)

    print("Max total words in citations:", max(entry_totals))
    print("Min total words in citations:", min(entry_totals))

    plt.figure(figsize=(8, 5))
    sns.histplot(entry_totals, bins=30, kde=True)
    plt.xlabel("Total words in all citations (per entry)")
    plt.ylabel("Count")
    plt.title("Distribution of total citation word count per entry")
    plt.tight_layout()
    plt.show()

def condense_and_filter_json_file(input_path, output_path, word_limit=3300):
    """
    Condenses abstracts in each citation to key sections and filters entries 
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

    # Load and process
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = []
    for entry in data:
        citations = entry.get("citations", {}) #get citations dict
        for k, citation in citations.items():
            abstract = citation.get("abstract")
            if abstract and str(abstract).strip():
                citation["abstract"] = extract_sections(abstract) #extract key sections from each abstract

        # Due to computational limits, we filter out entries where the total word count of all abstracts exceeds the limit.
        # This is a heuristic to ensure we do not overload the model. Only a few entries will be filtered out.
        total_words = sum(
            len(c["abstract"].split())
            for c in citations.values()
            if isinstance(c, dict) and c.get("abstract")
        )

        if total_words <= word_limit:
            filtered.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"Kept {len(filtered)} entries (<= {word_limit} words); removed {len(data) - len(filtered)}.")
    return filtered

if __name__ == "__main__":
    INPUT_DIR = "" #pdf storage directory with Apendix PDFs of ACR papers
    OUTPUT_DIR = "" #output directory for JSON files
    MERGED_OUTPUT = "" #output name for merged JSON
    CONDENSED_OUTPUT = "" #output name for condensed JSON with key sections and filtered entries
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_pdfs()
    merged = merge_all_outputs(MERGED_OUTPUT)
    condense = True  # Set to False if you want to skip condensing
    if condense:
        filtered = condense_and_filter_json_file(MERGED_OUTPUT, CONDENSED_OUTPUT)
        analyze_citation_lengths(filtered)
    else:
        analyze_citation_lengths(merged)


