import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from reward_functions import (
    format_reward,
    correctness_answer,
    llm_eval_reward,
    custom_embedding_reward
)

from reasoning_agent import Reasoning_Agent
from trl import GRPOConfig, GRPOTrainer

# Map reward function names to actual function references
REWARD_FUNCTIONS = {
    "format": format_reward,
    "answer": correctness_answer,
    "llm_eval": llm_eval_reward, #for llm-eval model
    "custom_embedding": custom_embedding_reward # for medreason-embed
}

def load_llm_eval_pipeline(model_id="Qwen/Qwen1.5-1.8B-Chat"):
    """Load the LLM evaluation pipeline if the 'llm_eval' reward is selected."""
    print("Loading LLM evaluator...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    pipeline_eval = pipeline("text-generation", model=model, tokenizer=tokenizer)
    print("LLM evaluation model loaded.")
    return pipeline_eval

def load_sbert():
    """Load the SBERT model if the 'custom embedding' reward is selected."""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2") #is it this?
    print("SBERT model loaded.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train Reasoning Agent with selected reward functions")
    parser.add_argument("--rewards", type=str, required=True,
                        help="Comma-separated list of rewards to use: format, answer, llm_eval, custom_embedding")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to JSON dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save model outputs")
    args = parser.parse_args()

    selected_rewards = [r.strip() for r in args.rewards.split(",")]
    print(f"Debugging: Selected reward functions: {selected_rewards}")

    # Load only the components required for selected reward functions
    if "llm_eval" in selected_rewards:
        llm_eval_pipeline = load_llm_eval_pipeline()

    if "custom_embedding" in selected_rewards:
        sbert = load_sbert()

    # Initialize model
    reasoning_agent = Reasoning_Agent()
    print("Reasoning Agent initialized.")

    train_dataset = reasoning_agent.prepare_dataset(args.data_path)

    # Compose reward function for the agent (different inputs to each reward function)
    reward_funcs = []
    if "answer" in selected_rewards:
        reward_funcs.append(("answer", correctness_answer))

    if "format" in selected_rewards:
        reward_funcs.append(("format", format_reward))

    if "llm_eval" in selected_rewards:
        reward_funcs.append((
            "llm_eval",
            lambda p, c: llm_eval_reward(
                p, c, dataset=train_dataset, llm_eval_pipeline=llm_eval_pipeline
            )
        ))

    if "custom_embedding" in selected_rewards:
        reward_funcs.append((
            "custom_embedding",
            lambda p, c: custom_embedding_reward(
                p, c, dataset=train_dataset, sbert=sbert
            )
        ))
    
    # Training configuration used for all the models
    training_args = GRPOConfig(
        learning_rate=5e-6, 
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=6, # Adjust based on your GPU memory
        gradient_accumulation_steps=2, # for smooth training
        num_generations=6, # number of generated responses per step
        max_prompt_length=4500,
        max_completion_length=1000, # reasoning trace + answer max length
        max_steps=200, #this equals approx 2 epochs with the current dataset size
        save_steps=100,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=args.output_dir,
    )

    # Initialize trainer with the reward function
    trainer = GRPOTrainer(
        model=reasoning_agent.model,
        tokenizer=reasoning_agent.tokenizer,
        train_dataset=train_dataset,
        reward_fn=reward_funcs,
        args=training_args,
        # Add any other necessary arguments
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()

    # an example how to run with two rewards:
    # python train_rl_reasoning_agent.py --rewards answer,format --data_path data/train.json --output_dir outputs
