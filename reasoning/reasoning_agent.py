from unsloth import FastLanguageModel 
import torch
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams 
import os
import json
from datasets import Dataset
from sklearn.model_selection import train_test_split
import tqdm


class Reasoning_Agent():
    def __init__(self, model_name="unsloth/Meta-Llama-3.1-8B-Instruct" , device="cuda:0"):
        max_seq_length = 5500  # context window as we will be adding abstracts
        lora_rank = 32 # The rank for LoRA adaptation matrices ( higher = slower but more accurate )

        #load model 
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            load_in_4bit = True, # quantization to 4 bits, faster
            fast_inference = True, # enable vLLM fast inference
            max_lora_rank = lora_rank,
            gpu_memory_utilization = 0.6, 
            device_map="auto"  #for gpu isage
        )

        print("Finished loading model")

        # Prepare Model for LoRA Fine-Tuning -> only some parameters will be trained
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = lora_rank, 
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ], 
            lora_alpha = lora_rank,
            use_gradient_checkpointing = "unsloth", 
            random_state = 42, # for reproducibility
        )

        print("Finished preparing model for LoRA fine-tuning")

    def prepare_dataset(self, json_path_to_data: str):
        """ Prepare the dataset for training and testing. It will be in the format of a list of dictionaries with 'prompt' and 'answer' keys."""

        SYSTEM_PROMPT_ABSTRACTS = """
        You are a medical AI assistant. Given ONLY the provided clinical condition, variant, medical procedure, and the following set of literature abstracts below, reason whether the procedure is, may be, or is not appropriate for this exact scenario.
        You should use the abstracts for evidence, referencing study findings, outcomes, and reported metrics only whenever they are directly relevant to the scenario. 
        When citing evidence from an abstract, refer to it by its reference number in square brackets (e.g., [7]).
        Ignore abstracts or abstract sections that do not pertain to the current case; include only information that meaningfully supports your reasoning. 
        If there is no clinical evidence based on the abstracts, respond with: "There is no relevant clinical literature for this variant and procedure."


        Think through the question step by step. Answer in clear, self-contained sentences, such that each sentence can stand alone as a **reasoning trace** for this clinical decision.

        Respond in the following XML format:
        <think>
        [Your step-by-step reasoning here; reason ONLY about the given clinical scenario, citing relevant findings or statistics from the provided abstracts if appropriate. Do NOT discuss other cases, diseases, or variants.]
        </think>
        <answer>
        [One of: Usually Appropriate, May Be Appropriate, Usually Not Appropriate]
        </answer>

        DO NOT provide information about any other condition, variant, or procedure.
        DO NOT fabricate or reference evidence that is not present in the provided abstracts.
        """

        # make the user content for the prompt
        def make_user_content(condition, variant, medical_procedure, abstracts):
            return (
                f"Clinical condition: {condition}\n"
                f"Variant: {variant}\n"
                f"Procedure: {medical_procedure}\n"
                f"Abstracts: {abstracts}"  # remove this if you want to use the prompt without citations (Baseline model)
            )

        # open the json file and load the data
        with open(json_path_to_data, 'r', encoding='utf-8') as f:
            data = json.load(f)

        processed_data = []
        for item in data:
            # create the prompt with the format that LLama model expects
            prompt = [
                {'role': 'system', 'content': SYSTEM_PROMPT_ABSTRACTS}, 
                {'role': 'user', 'content': make_user_content(item['category'], item['variant'], item['procedure'], item['citations'])}
            ]
            appropriateness = item['appropriateness']
            acr_traces = item.get('reasoning_traces', "")  # in case empty

            # join list into a string of sentences of reasoning traces.
            if isinstance(acr_traces, list):
                reasoning = " ".join(str(x) for x in acr_traces)  
            elif isinstance(acr_traces, str): 
                reasoning = acr_traces
            else:
                reasoning = ""

            answer = f"<think>{reasoning}</think>\n<answer>{appropriateness}</answer>"
            processed_data.append({'prompt': prompt, 'answer': answer}) #final prompt format

        return processed_data

    def run_inference(self, test_dataset, lora_path: str, max_new_tokens: int = 1024):
        """ Run inference on the test dataset using the model with LoRA weights. 
        You can find the LoRA weights in the Huggingface repository (annietz/grpo-acr-adapters) and download them.
        After you store them locally at a folder, you can pass the path to the folder as lora_path.
        The test_dataset should be a list of dictionaries with 'prompt' and 'answer' keys using the prepare_dataset function."""
        
        # Load LoRA weights
        lora = self.model.load_lora(lora_path)
        print("LoRA weights loaded from:", lora_path)

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=max_new_tokens,
        )

        outputs = []

        # Iterate over the test dataset -> You can also do this in batches 
        for example in tqdm(test_dataset):
            prompt = example['prompt']
            text = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Generate model output
            result = self.model.fast_generate(
                text,
                sampling_params=sampling_params,
                lora_request=lora,
            )

        # If result is a list of outputs, take the first
        generated = result[0].outputs[0].text if hasattr(result[0], 'outputs') else result[0].text

        outputs.append({
            "input": text,
            "output": generated,
            "gold_answer": example['answer']
        })
        print(f"Generated {len(outputs)} outputs.") #this file we also do process on more!
        
        # Post-process for readability: extract scenario, variant, procedure,  reasoning and answer and make json file
        parsed_results = []
        for row in outputs:
            input_text = row["input"]
            user_block = re.search(r"user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>", input_text, re.DOTALL)
            user_block = user_block.group(1) if user_block else ""
            lines = user_block.split('\n')

            clinical_scenario = variant = procedure = ""
            for line in lines:
                if line.startswith("Clinical condition:"):
                    clinical_scenario = line.replace("Clinical condition:", "").strip()
                elif line.startswith("Variant:"):
                    variant = line.replace("Variant:", "").strip()
                elif line.startswith("Procedure:"):
                    procedure = line.replace("Procedure:", "").strip()

            def extract_think_answer(text):
                think = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
                answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
                return (think.group(1).strip() if think else ""), (answer.group(1).strip() if answer else "")

            llm_reasoning, llm_answer = extract_think_answer(row["output"])
            gold_reasoning, gold_answer = extract_think_answer(row["gold_answer"])

            parsed_results.append({
                "clinical_scenario": clinical_scenario,
                "variant": variant,
                "procedure": procedure,
                "llm_reasoning": llm_reasoning,
                "llm_answer": llm_answer,
                "gold_reasoning": gold_reasoning,
                "gold_answer": gold_answer
            })

        return parsed_results
    

# Note: Use this prompt if you want to use the model without citations : Baseline

#SYSTEM_PROMPT = """
#You are a medical AI assistant. Given ONLY the provided clinical condition, variant, and medical procedure below, reason whether the procedure is, may be, or is not appropriate for this exact scenario.
#Think through the question step by step. Answer in clear, self-contained sentences, such that each sentence can stand alone as a **reasoning trace** for this clinical decision.

#Respond in the following XML format:
#<think>
#[Your step-by-step reasoning here; ONLY about the clinical scenario given, and do NOT discuss other cases, diseases, or variants.]
#</think>
#<answer>
#[One of: Usually Appropriate, May Be Appropriate, Usually Not Appropriate]
#</answer>

#DO NOT provide information about any other condition, variant, or procedure.
#"""