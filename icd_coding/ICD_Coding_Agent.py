import transformers
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import string
import json
from typing import List, Dict
from dotenv import load_dotenv
import os
import pandas as pd


class ICD_Coding_Agent:
    """
    Agent for normalizing (Italian) clinical terms to standardized English ICD-9-CM codes.
    The pipeline consists of:
    1. Using an LLM model to convert noisy clinical descriptions into concise ICD-9-CM - like texts.
    2. Using a SentenceTransformer model to encode the transformed, standardized query.
    3. Searching a FAISS index (already implemented) for similar ICD codes.
    4. Reranking results using a LLM model for better accuracy.
    """

    def __init__(self,
                 embedding_model_name: str = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
                 llama_model_id: str = "meta-llama/Llama-3.1-8B",
                 faiss_index_path: str = None, #where the index is stored
                 codes_path: str = None, #json file that connects ICD codes to their descriptions
                 device: str = None):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model_name)

        # Load model via HuggingFace transformers pipeline
        load_dotenv()
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN") or "" # replace with your actual token

        # initialize llm pipeline (will be use for standardization and reranking)
        self.llm_pipeline = transformers.pipeline(
            "text-generation",
            model=llama_model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            use_auth_token=self.hf_token
        )

        # Load FAISS index and ICD codes json
        self.index = faiss.read_index(faiss_index_path) if faiss_index_path else None
        with open(codes_path, "r", encoding="utf-8") as f:
            self.codes = [json.loads(line) for line in f]

        print("ICD_Coder_Agent initialized successfully.")


    def normalize_clinical_term(self, clinical_note: str) -> str:
        """
        Normalize a noisy Italian clinical term into a ICD compatible vocabulary medical term in English.
        """

        prompt = f"""You are a clinical terminology normalization assistant. Your task is to convert noisy,  unstructured Italian clinical diagnosis descriptions into concise, standardized English medical terms suitable for ICD-9-CM matching.
            Rules:
            - Do NOT translate word-for-word but convert to a medically accurate, concise ICD-09-CM terms.
            - Return just the core medical concept in English and use standard medical terminology.
            - Avoid unnecessary details or qualifiers.
            - DO NOT add comments, notes, or any extra information.
            - NEVER include gender, age, or location.

            Format:
            Input: [Italian clinical description]
            Output: [English medical diagnosis terms]

            Examples:
            Input: "mallatia renale cronica statio finale"
            Output: "Chronic kidney disease, Stage V"

            Input: "Neoplasia maligna della mammella femminile"
            Output: "Malignant neoplasm of female breast, unspecified"

        Input: "{clinical_note.strip()}"
        Output:"""

        # Employ the llm_pipeline
        generation = self.llm_pipeline(
            prompt,
            max_new_tokens=60, #limit the output length to avoid excessive text
            do_sample=False,
            temperature=0.1, # low temperature for deterministic output 
            return_full_text=False
        )
        
        # extract the generated text
        generated_text = generation[0]["generated_text"]
        diagnosis_text= generated_text.split("Output:")[-1].split("\n")[0].strip()
        return diagnosis_text

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for ICD codes similar to the standardized query using FAISS index.
        Input:
        - query: A normalized clinical term in English.
        - top_k: Number of top results to return.
        Output:
        - A list of dictionaries containing ICD codes and their descriptions, sorted by relevance.
        """

        #embed the given STANDARDIZED clinical note
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(query_vec, top_k) # search the index and get the top_k results

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.codes):
                item = self.codes[idx].copy() ##this copies the code,description
                item["score"] = float(score)
                results.append(item)

        scores = [r["score"] for r in results]
        
        # Check if results are of low confidence
        # this is defined as the top result or the variance between retrieved scores is below a threshold
        # If they have low confidence, we rerank them using the LLM
        if results and (scores[0] < 0.70 and np.var(scores) < 0.0005):
            print(f"[INFO] Low confidence in results, with variance")
            return self.llm_rerank(query, results)
        
        return results
    
    def llm_rerank(self, query: str, results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        prompt_text = (
            "Rank the ICD codes below by relevance to the QUERY, from most relevant to least relevant. \n"
            "Return ALL of the candidate ICD codes exactly once, in order of relevance.\n"
            "Output ONLY the list of ICD codes, comma-separated, NO explanations or extra text.\n"
            "QUERY: \n"
            f"{query}\n"
            "Candidates:\n" +
            "\n".join([f"{r['code']}: {r['description']}" for r in results]) +
            "\n\n"
            "Answer:"
        )

        # Use the llm_pipeline for the reranking
        output = self.llm_pipeline(
            prompt_text,
            return_full_text=False,
            max_new_tokens=100,
            temperature=0.1, # low temperature for deterministic output
            top_p=0.9,
            repetition_penalty=1.1,
        )

        # Extract the generated text from the output
        answer_text = output[0]["generated_text"].strip()
        if "Answer:" in answer_text:
            answer_text = answer_text.split("Answer:")[-1].strip() #clean further if needed
    
        codes = [c.strip() for c in answer_text.split(",") if c.strip()] #find all the codes
        code_map = {r["code"]: r for r in results} #map them back to their descriptions
        reranked = list(code_map.values()) #return in a list form for compatibility

        # avoid empty reranked lists, return original list
        if not reranked:
            print("[WARNING] LLM reranker output invalid or empty. Returning original results.")
            return results

        return reranked


    def evaluate(self, data):
        """ Evaluate the model's performance on a dataset. 
        This is to be done for our own dataset where we had a dataframe of clinical notes and we added a column with the "top_5_predicted_codes by leveraging the above pipeline
        Importantly, we also have the ground truth codes for each clinical note (column "code")
        """

        def in_top_k(true_code: str, predicted_list: List[str], k: int = 5) -> bool:
            true_code = str(true_code)
            return true_code in [str(c) for c in predicted_list[:k]]

        def reciprocal_rank(true_code, predictions):
            true_code = str(true_code)
            try:
                return 1 / (predictions.index(true_code) + 1)
            except ValueError:
                return 0

        top_5_accuracy = sum(
            in_top_k(row["code"], row["top_5_predicted_codes"], 5)
            for _, row in data.iterrows()
        ) / len(data)

        top_1_accuracy = sum(
            str(row["top_5_predicted_codes"][0]) == str(row["code"])
            for _, row in data.iterrows()
        ) / len(data)

        # To evaluate whether the key condition (first 3 digits match) -> leverage how ICD are structured.
        top_1_hierarchical = sum(
            str(row["top_5_predicted_codes"][0])[:3] == str(row["code"])[:3]
            for _, row in data.iterrows()
        ) / len(data)

        mrr_score = sum(
            reciprocal_rank(row["code"], [str(c) for c in row["top_5_predicted_codes"]])
            for _, row in data.iterrows()
        ) / len(data)

        print(f"Top-1 Accuracy: {top_1_accuracy:.2%}")
        print(f"Top-5 Accuracy: {top_5_accuracy:.2%}")
        print(f"Top-1 Hierarchical Accuracy: {top_1_hierarchical:.2%}")
        print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")

if __name__ == "__main__":

    # Example usage
    icd_coding_agent = ICD_Coding_Agent(
        embedding_model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        llama_model_id="meta-llama/Llama-3.1-8B",
        faiss_index_path="data/index/icd_index.faiss", #where the index is stored
        codes_path="data/index/icd_codes.json" #where the codes are stored in the form {code:, description:}
    )

    # If you want to see a walk-through example of the pipeline.
    #query = "tumori maligni della prostata"
    #normalized_term = icd_coding_agent.normalize_clinical_term(query)
    #print(f"Normalized Term: {normalized_term}")

    #results = icd_coding_agent.search(normalized_term, top_k=5)
    #print("Search Results:", results)
