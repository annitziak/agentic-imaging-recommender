
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util

class ACRCriteriaChecker:
    """
    The ACRCriteriaChecker aims to see if the ICD code assigned to a clinical note is matched with one of the ICD
    codes in the ACR Criteria that were used for training. If no such code exists then the Medical Review Agent is enabled.
    We already have a dataset of ACR conditions with their corresponding ICD codes (these were mapped offline using the ICD_Coding_Agent), and we will use this to check for matches.
    Notably this custom function also aims to connect the clinical note also to the correct variant by collecting extra information
    such as Age, Anamnesis and otherwise enables an embedding based method, defaulting to variant 1, if no specific variant is found.
    """
    def __init__(self, acr_df, model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"):
        self.df = acr_df # this df contains for each ACR variant a corresponding ICD code and its description
        self.model = SentenceTransformer(model_name) # used for matching in the variant level
        print("Loaded ACR Criteria Checker with embedding model")

    def split_into_sentences(self, text):
        """Simple helper function. Splits text into sentences based on periods."""
        return [s.strip() for s in text.split('.') if s.strip()]
    

    def max_sentence_similarity(self, text1, text2):
        """Calculate the maximum cosine similarity between sentences in two texts. This is to best identify a match."""
        sents1 = self.split_into_sentences(text1)
        sents2 = self.split_into_sentences(text2)
        if not sents1 or not sents2:
            return 0.0
        embeds1 = self.model.encode(sents1, convert_to_tensor=True)
        embeds2 = self.model.encode(sents2, convert_to_tensor=True)
        sim_matrix = util.pytorch_cos_sim(embeds1, embeds2)

        # return the maximum similarity score
        return sim_matrix.max().item()

    def matching_variant(self, clinical_note, icd):
        """Find the best matching condition variant for a given ICD code and clinical note."""

        # Firstly find if the ICD code exists is covered by our ACR scraped criteria 
        filtered_df = self.df[self.df['ICD'] == icd]
        if filtered_df.empty:
            return None # if no matching ICD code, return None -> Medical Review Agent will be employed


        # If the ICD is in our ACR Conditions, try to narrow down the variants based on the clinical note

        # First: check for specific AGE in the clinical note
        age_match = re.search(r'[FMfm]:?\s*(\d{1,3})', clinical_note)
        print("Age Match:", age_match)
        if age_match:
            age = int(age_match.group(1))
            age_variants = filtered_df[filtered_df['Condition_Variant'].str.contains(r'\bAge\b|\bAged\b', case=False, na=False)]
            for _, row in age_variants.iterrows():
                variant_text = row['Condition_Variant']
                # Check for specific age conditions in the variant text. This is standard terminology in the ACR docs.
                # Check if less than:
                less_than = re.search(r'Age\s+less than\s+(\d+)', variant_text, re.IGNORECASE)
                if less_than and age < int(less_than.group(1)):
                    return variant_text
                # Check if from ... to ...
                range_match = re.search(r'Age\s+(\d+)\s*to\s*(\d+)', variant_text, re.IGNORECASE)
                if range_match:
                    low, high = int(range_match.group(1)), int(range_match.group(2))
                    if low <= age <= high:
                        return variant_text
                # Check if greater than:
                gte_match = re.search(r'Age\s+greater than or equal to\s+(\d+)', variant_text, re.IGNORECASE)
                if gte_match and age >= int(gte_match.group(1)):
                    return variant_text
                
            # If not age match but age is given, return variant 1.   
            variant_1 = filtered_df[filtered_df['Condition_Variant'].str.contains(r'Variant\s*1\b', case=False, na=False)]
            if not variant_1.empty: #fallback if not found.
                return variant_1.iloc[0]['Condition_Variant']

        # Secondly, search for specific anamnesis phrases in the clinical note.
        # These are some common used in ACR criteria.
        anamnesis_phrases = ["Initial imaging", "Next imaging study", "Pretreatment evaluation", "Follow-up imaging"]
        matched_variants = [
            variant_text
            for variant_text in filtered_df['Condition_Variant']
            if any(
                phrase.lower() in variant_text.lower() and phrase.lower() in clinical_note.lower()
                for phrase in anamnesis_phrases)]

        if matched_variants:
            matches = filtered_df[
                filtered_df['Condition_Variant'].apply(
                    lambda x: any(
                        phrase.lower() in x.lower() and phrase.lower() in clinical_note.lower()
                        for phrase in anamnesis_phrases))]
            # If matches found, rank them by similarity and return the best one
            if not matches.empty:
                best_variant = None
                best_score = -1
                for _, row in matches.iterrows():
                    sim = self.max_sentence_similarity(clinical_note, row['Condition_Variant'])
                    if sim > best_score:
                        best_score = sim
                        best_variant = row['Condition_Variant']
                return best_variant

        # If nothing found so far, we will try to find the best matching variant based on sentence similarity
        best_variant = None
        best_score = -1
        for _, row in filtered_df.iterrows():
            sim = self.max_sentence_similarity(clinical_note, row['Condition_Variant'])
            if sim > best_score:
                best_score = sim
                best_variant = row['Condition_Variant']

        # If the best score is above a threshold, return the that variant -> MATCH
        # This is to make sure that we have a confident match
        if best_score > 0.5:
            return best_variant

        # If no specific variant found, return the first variant or variant 1 as a fallback
        variant_1 = filtered_df[filtered_df['Condition_Variant'].str.contains(r'Variant\s*1\b', case=False, na=False)]
        if not variant_1.empty:
            return variant_1.iloc[0]['Condition_Variant']
        
        # If no variant 1 found, return the first condition variant found
        return filtered_df.iloc[0]['Condition_Variant']

if __name__ == "__main__":

    # This is the static dataset created offline using our ICD coding pipeline and human evaluation
    # It has columns : Condition_Variant, ICD and LONG DESCRIPTION of the code

    df = pd.read_excel("data/ACR_ICD_Mapping.xlsx")

    # Initialize the agent
    checker = ACRCriteriaChecker(df)

    # Example to match a clinical note and icd to a variant. 
    # Giving extra age information to see the matching logic in action.
    # In the form F39, where F means Female and 39 means age 39.
    variant = checker.matching_variant("Adult with significant chest pain. F39", "61171")
    #  Should match with -> Breast Pain. Variant 3: Female with clinically significant breast pain (focal and noncyclical). Age 30 to 39. Initial imaging.
    print("Matched Variant:", variant)