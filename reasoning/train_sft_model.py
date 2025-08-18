from unsloth import FastLanguageModel, is_bfloat16_supported
import json
import torch
import re
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

# Configuration
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"
max_seq_length = 5500
lora_rank = 32 # increase rank for better performance

# Load model using FastLanguageModel with LoRA support
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True, # memory efficient loading
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
    device_map="auto" 
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # LoRA target modules
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=42
)

print("Loaded model and LoRA PEFT model.")

# System prompt for the SFT model
SYSTEM_PROMPT = """
You are a medical AI assistant. Given ONLY the provided clinical condition, variant, and medical procedure below, respond with one of:
Usually Appropriate, May Be Appropriate, Usually Not Appropriate.

DO NOT provide any other information.
"""

def make_user_content(condition, variant, medical_procedure):
    return (
        f"Clinical condition: {condition}\n"
        f"Variant: {variant}\n"
        f"Procedure: {medical_procedure}\n"
        "Answer:"
    )

# Load and process the ACR train dataset
def process_acr_dataset(json_path, eos_token=""):
    """ Processes the ACR dataset from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        condition = item['condition']
        variant = item['variant']
        procedure = item['procedure']
        appropriateness = item['appropriateness']

        # Prompt components
        system_prompt = SYSTEM_PROMPT.strip()
        user_prompt = make_user_content(condition, variant, procedure)
        answer = f"<answer>{appropriateness}</answer>"

        full_text = f"{system_prompt}\n{user_prompt}\n{answer}{eos_token}"
        processed_data.append({"text": full_text})

    return Dataset.from_list(processed_data)

json_path = ""  # Path to your training ACR dataset JSON file
training_dataset = process_acr_dataset(json_path)

print("Example training sample and format:\n", training_dataset[0]['text'])

# Training params and setup
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        max_steps=300,  #  this is the same as 2 epochs of the RL model for fair comparison
        save_steps=100,
        learning_rate=5e-6, #same learning rate
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.1,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="outputs_sft",
        report_to="none",
    ),
)

print("Loaded params and starting training...")
trainer_stats = trainer.train()

# Saving the trained model and tokenizer
model.save_pretrained("sft_model")
tokenizer.save_pretrained("sft_model")
print("Model and tokenizer saved to `sft_model`.")

