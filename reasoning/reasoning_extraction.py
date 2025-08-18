import os
import json
import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model and tokenizer loading
MODEL_NAME = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN") or "your_huggingface_token_here"  # replace with your Hugging Face token

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    token=HF_TOKEN
)
print("Model and tokenizer loaded successfully.")

def extract_reasoning_traces(procedure: str, variant: str, reasoning: str) -> List[str]:
    prompt = f"""Explain the clinical reasoning behind the decision to recommend {procedure} for {variant}.
Break the provided reasoning text into a sequence of concise reasoning steps, preserving the original order and logic.
Each step should represent a distinct point or idea from the original text.
Focus especially on steps that reflect expert consensus or cite specific studies (e.g., [number]).
Keep each step brief and focused.
Keep the citations  (e.g., [number]).
Output only the reasoning steps as a numbered list; do not add any labels, summary, or extra information.

Reasoning:
{reasoning}
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2, # low for deterministic output
            top_p=0.1,
            repetition_penalty=0.0,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    traces = []
    for line in decoded.split("\n"):
        line = line.strip()
        if line.startswith(tuple(f"{i}." for i in range(1, 100))):
            traces.append(line.split(".", 1)[1].strip())

    if not traces:
        traces = [decoded.strip()]
    return traces


def process_json_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for _, entry in enumerate(data):
        reasoning = entry.get("reasoning", "") # Default to empty string if not present
        procedure = entry.get("procedure", "[MODALITY]")
        variant = entry.get("condition", "[CLINICAL SCENARIO / VARIANT]")
        #print(f"Processing entry {i + 1}/{len(data)}: {procedure} - {variant}")
        traces = extract_reasoning_traces(procedure, variant, reasoning)
        entry["reasoning_traces"] = traces

    # Save the modified data back to a new JSON file with traces
    output_path = filepath.replace(".json", "_with_traces.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")


#if __name__ == "__main__":
    # Replace with your actual file name that contains the JSON data in the form:
    # {condition: "condition", procedure: "procedure", reasoning: "reasoning"}
    #process_json_file("your_file.json")