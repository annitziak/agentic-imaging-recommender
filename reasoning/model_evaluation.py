import argparse
import os
import re
import json
import numpy as np
import torch
import transformers
import string
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

# This script evaluates the performance using three key metrics:
# 1. Performance scores :  F1 scores and Confusion Matrix -> function evaluate_classification
# 2. NER embedding F1 score -> function evaluate_entity_embedding_alignment
# 3. LLM alignment score (0-10 scale) -> function evaluate_llm_alignment


# Helper function
def process_label(label: str) -> str:
    """Process the label to standardize it for evaluation."""
    label = label.strip().lower().replace('(', '').replace(')', '')
    if "usually appropriate" in label:
        return "usually appropriate"
    elif "may be appropriate" in label:
        return "may be appropriate"
    elif "usually not appropriate" in label:
        return "usually not appropriate"
    return "unknown"

# Classification Evaluation
def evaluate_classification(data):
    """Evaluate classification performance using confusion matrix and F1 scores."""
    gold_labels = [process_label(item['gold_answer']) for item in data]
    pred_labels = [process_label(item['llm_answer']) for item in data]

    # Plot confusion matrix
    labels_order = ["usually not appropriate", "may be appropriate", "usually appropriate"]
    cm = confusion_matrix(gold_labels, pred_labels, labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix: LLM Answer vs. Gold Standard")
    plt.xlabel("LLM Answer")
    plt.ylabel("Gold Answer")
    plt.xticks(rotation=30)
    #plt.savefig("figures/confusion_matrix.png", bbox_inches='tight')
    plt.show()

    report = classification_report(gold_labels, pred_labels, labels=labels_order, digits=3)
    print(report)
    return report

# NER Alignment Evaluation
def evaluate_entity_embedding_alignment(data, NER_MODEL_NAME, SBERT_MODEL_NAME):
    recalls, precisions = [], []

    # also add a regex to match phrases indicating no relevant literature 
    # these phrases are common but they would not be captured by the clinical NER model
    PHRASES = [
        "no relevant clinical literature", "no relevant literature found", "no relevant evidence found",
        "no relevant studies found", "no evidence available", "no relevant studies available", "literature"
    ]
    PHRASE_ENTITY = "no relevant literature found"
    PHRASE_REGEXES = [re.compile(re.escape(p), re.IGNORECASE) for p in PHRASES]

    def extract_entities(text: str):
        """Extract entities from text using NER and regex."""
        entities = []
        for rx in PHRASE_REGEXES:
            if rx.search(text):
                entities.append(PHRASE_ENTITY) #append the same PHRASE_ENTITY if any of the regex match (to ensure match with similarity after)
                break
        # extract entities using NER pipeline 
        results = ner_pipeline(text[:1500]) # limit to prevent OOM errors
        entities.extend([
            ent['word'].lower().translate(str.maketrans('', '', string.punctuation)).strip()
            for ent in results
        ])
        return list(set(entities)) # # remove duplicates

    def entity_embeddings(entities):
        """Get embeddings for a list of entities using SBERT."""
        if not entities:
            return torch.empty(0, sbert.get_sentence_embedding_dimension()) # return empty tensor if no entities
        return sbert.encode(entities, convert_to_tensor=True)

    # Load the NER model and SBERT embeddings
    ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, grouped_entities=True, token=os.getenv("HUGGINGFACE_TOKEN"))
    sbert = SentenceTransformer(SBERT_MODEL_NAME)

    print("Loaded NER model and SBERT embeddings.")

    # Process each item in the dataset
    for idx, item in enumerate(data):
        gold_ents = extract_entities(item.get("gold_reasoning", ""))
        gen_ents = extract_entities(item.get("llm_reasoning", ""))
        if not gold_ents or not gen_ents: # skip if no entities found
            #print(f"Skipping item {idx} due to missing entities: gold={gold_ents}, gen={gen_ents}")
            continue

        # Get embeddings for both sets of entities and create similarity matrix to compute recall and precision
        gold_emb = entity_embeddings(gold_ents)
        gen_emb = entity_embeddings(gen_ents)

        cosine_sim_matrix = torch.mm(gold_emb, gen_emb.T)
        gold_norm = gold_emb.norm(dim=1)[:, None]
        gen_norm = gen_emb.norm(dim=1)[None, :]
        cosine_sim_matrix = cosine_sim_matrix / (gold_norm * gen_norm + 1e-8)

        # Recall: max similarity to any generated entity
        max_sim_gold = cosine_sim_matrix.max(dim=1).values
        mean_recall = max_sim_gold.mean().item()

        # Precision: max similarity to any gold entity
        max_sim_gen = cosine_sim_matrix.max(dim=0).values
        mean_precision = max_sim_gen.mean().item()

        recalls.append(mean_recall)
        precisions.append(mean_precision)

    recall_mean = np.mean(recalls)
    precision_mean = np.mean(precisions)
    # Calculate F1 score
    f1 = 2 * recall_mean * precision_mean / (recall_mean + precision_mean + 1e-8)
    return {"mean_recall": recall_mean, "mean_precision": precision_mean, "f1": f1}


# LLM Alignment Evaluation
def make_alignment_prompt(llm_reasoning, gold_reasoning):
    return (
        "You are an expert medical evaluator. Evaluate the alignment between the following two clinical reasonings on a scale from 0 (not aligned) to 10 (perfectly aligned).\n"
        "Consider these aspects carefully:\n"
        "- **Semantic Coverage-Step**: Does the model’s reasoning capture the essential semantic elements and clinical concepts present in the gold reasoning?\n"
        "- **Reasoning Alignment**: How well does the sequence and content of reasoning steps align with the ground-truth chain?\n"
        "- **Logical Process**: Is the logical flow of the reasoning coherent, valid, and medically sound?\n"
        "- **Faithfulness**: Are the details and claims accurate and consistent with the source reasoning?\n\n"
        "Output only a single integer number from 0 to 10 representing the overall alignment score. Do NOT include any explanations, comments, or additional text.\n\n"
        "Be very concise and strict and consider the aspects! \n"
        f"Model Reasoning:\n{llm_reasoning}\n\n"
        f"Reference (Gold) Reasoning:\n{gold_reasoning}\n\n"
        "Alignment score:"
    )

def extract_score(text):
    """Extract the alignment score from the LLM output text."""
    match = re.search(r"Alignment score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if not match:
        match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", text)
    if match:
        score = float(match.group(1))
        return int(score) if score.is_integer() else score
    return None

def evaluate_llm_alignment(data):
    """Evaluate LLM reasoning alignment using a pre-trained LLM model."""

    # Load the LLM model and tokenizer
    LLM_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, token=os.getenv("HUGGINGFACE_TOKEN"))
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        token=os.getenv("HUGGINGFACE_TOKEN")
    )

    print("Loaded standard Hugging Face model for alignment evaluation.")
    print("Loaded LLM model for alignment evaluation.")
    scores = []
    for item in tqdm(data):
        prompt = make_alignment_prompt(item["llm_reasoning"], item["gold_reasoning"])
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
        output = llm_model.generate(**inputs, max_new_tokens=5, do_sample=False)
        decoded = llm_tokenizer.decode(output[0], skip_special_tokens=True)
        score = extract_score(decoded)
        if score is not None:
            scores.append(score)
    return {"mean_alignment_score": np.mean(scores), "all_scores": scores}

# Run the evaluation script
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Evaluate LLM model outputs.")
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the evaluation JSON file")
    parser.add_argument("--classification", action="store_true",
                        help="Run classification metrics (F1, confusion matrix.)")
    parser.add_argument("--entities", action="store_true",
                        help="Run entity embedding evaluation (NER + SBERT)")
    parser.add_argument("--llm", action="store_true",
                        help="Run LLM-based reasoning alignment evaluation")
    args = parser.parse_args()

    with open(args.path, "r") as f:
        data = json.load(f)

    if args.classification:
        print("\n Running Classification Evaluation...")
        report = evaluate_classification(data)

    if args.entities:
        NER_MODEL_NAME = "OpenMed/OpenMed-NER-PathologyDetect-PubMed-v2-109M"
        SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

        print("\nRunning Entity Embedding Evaluation...")
        entity_metrics = evaluate_entity_embedding_alignment(data, NER_MODEL_NAME, SBERT_MODEL_NAME)
        print(f"\n Mean Recall:    {entity_metrics['mean_recall']:.3f}")
        print(f" Mean Precision: {entity_metrics['mean_precision']:.3f}")
        print(f" Entity F1 Score: {entity_metrics['f1']:.3f}")

    if args.llm:
        print("\n Running LLM Alignment Evaluation (0–10 scale)...")
        alignment_results = evaluate_llm_alignment(data)
        print(f"\n Mean LLM Alignment Score: {alignment_results['mean_alignment_score']:.2f}")


    # example run: python model_evaluation.py --path data/generalization_dataset_example.json --classification --entities --llm