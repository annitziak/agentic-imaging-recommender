import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer, util

#--- helper functions ---
def extract_xml_answer(text: str) -> str:
    """Extracts content from <answer>...</answer> block, handles missing/malformed tags, if any."""
    # Solid match for <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # If missing closing tag, extract everything after <answer>
    match = re.search(r"<answer>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        # Stop at the next tag if present
        content = re.split(r"<.*?>", content)[0]
        return content.strip()
    return ""

def extract_xml_think(text: str) -> str:
    """Extracts content from <think>...</think> block, handles missing/malformed tags."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    # Solid match for <think>...</think>
    if match:
        return match.group(1).strip()
    # If missing closing tag, extract everything after <think>
    match = re.search(r"<think>(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = re.split(r"<.*?>", match.group(1))[0]
        return content.strip()
    return ""

def standardize_label(label):
    label = label.strip().lower()
    # Only match the three main classes at the start of the string
    bases = {
        "usually appropriate": "Usually Appropriate",
        "may be appropriate": "May Be Appropriate",
        "usually not appropriate": "Usually Not Appropriate"
    }
    for base in bases:
        if label.startswith(base):
            return bases[base]
    return "Unknown"


#--------Rewards functions--------

# Answer reward function
def correctness_answer(prompts, completions, answer, **kwargs) -> list[float]:
    """Computes correctness reward based on exact match of generated answer with gold answer."""
    responses = []
    for completion in completions:
        # Accept both list-of-dicts or direct string/dict
        if isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict):
                content = completion[0].get('content', '')
            else:
                content = str(completion[0])
        elif isinstance(completion, dict):
            content = completion.get('content', '')
        else:
            content = str(completion)
        responses.append(content)

    # Extract and standardize model outputs and gold answers
    extracted_responses = [
        standardize_label(extract_xml_answer(r))
        for r in responses
    ]
    gold_answers = [
        standardize_label(extract_xml_answer(a))
        for a in answer
    ]
    # Compute reward: 1 if match, else 0 (as a list for generated responses)
    reward = [1.0 if r == a else 0.0 for r, a in zip(extracted_responses, gold_answers)]
    return reward


# Format reward function
def format_reward(prompts, completions, completion_ids=None, **kwargs):
    """
    Rewards completions that contain exactly one <think>...</think> and one <answer>...</answer> block (in that order).
    """
    rewards = []
    for completion in completions:
        # Flexible extraction for different data structures
        if isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict):
                text = completion[0].get('content', '')
            else:
                text = str(completion[0])
        elif isinstance(completion, dict):
            text = completion.get('content', '')
        else:
            text = str(completion)
        
        # Extract <think> and <answer> blocks (allow any whitespace in-between)
        think_matches = re.findall(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE)
        answer_matches = re.findall(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE)

        # Must be exactly one of each, in order (first think, then answer)
        if len(think_matches) != 1 or len(answer_matches) != 1:
            rewards.append(0.0)
            continue

        # Everything before, between, and after must be only whitespace, check with regex
        pattern = r"^\s*" + re.escape(think_matches[0]) + r"\s*" + re.escape(answer_matches[0]) + r"\s*$"
        if re.match(pattern, text, flags=re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


# --- LLM evaluation reward function ---

def llm_eval_reward(prompts, completions, dataset, llm_eval_pipeline, **kwargs) -> list[float]:
    def build_prompt(gold_reasoning, generated_reasoning):
        return (
            "Evaluate the alignment between the following two reasonings on a scale from 0 (not aligned) to 10 (perfectly aligned) for each of the dimensions below:\n"
            "1. Relevant medical concepts: mentioning of relavant medical terms.\n"
            "2. Reasoning logic alignment: presence of arguments for final recommendation.\n"
            "3. Use of supporting evidence: appropriate referencing and justification.\n\n"
            "Output ONLY a JSON object with integer scores for each dimension, with keys exactly as follows:\n"
            "{\"medical_concepts\": <int>, \"logic_alignment\": <int>, \"supporting_evidence\": <int>}\n"
            "Do not include any explanation or text besides this JSON.\n\n"
            f"Reasoning 1:\n{gold_reasoning}\n\n"
            f"Reasoning 2:\n{generated_reasoning}\n\n"
            "Scores:"
        )

    def parse_scores(output_text):
        """Extracts JSON scores from the LLM output. Returns them in the form of a dictionary."""
        json_candidates = re.findall(r'\{.*?\}', output_text, re.DOTALL)
        for candidate in reversed(json_candidates):
            try:
                scores = json.loads(candidate)
                if isinstance(scores, dict) and all(isinstance(k, str) and isinstance(v, int) for k, v in scores.items()):
                    return scores
            except json.JSONDecodeError:
                continue
        print(f"[WARN] Could not parse valid JSON scores from output:\n{output_text}")
        return None

    rewards = []

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Extract gold answer and reasoning from dataset
            gold_answer_and_reasoning = dataset[i].get("answer", "")
            gold_reasoning = extract_xml_think(gold_answer_and_reasoning)

            # Extract model-generated content
            if isinstance(completion, list) and len(completion) > 0:
                text = completion[0].get("content", "")
            elif isinstance(completion, dict):
                text = completion.get("content", "")
            else:
                text = str(completion)

            generated_reasoning = extract_xml_think(text) or text.strip()

            # Build the prompt for LLM evaluation contains gold and generated reasoning
            prompt_text = build_prompt(gold_reasoning, generated_reasoning)

            output = llm_eval_pipeline(
                prompt_text,
                max_new_tokens=100,
                do_sample=False,
                top_p=1.0,
                return_full_text=False
            )[0]["generated_text"]

            # Parse the output to extract scores
            scores = parse_scores(output)
            if scores:
                # Average the scores across the three dimensions
                avg_score = sum(scores.values()) / (10 * len(scores))
                rewards.append(avg_score)
            else:
                rewards.append(0.0)

        except Exception as e:
            print(f"[ERROR] LLM Eval failed at index {i}: {e}")
            rewards.append(0.0)

    return rewards


# Custom embedding reward for MedReason Embed model
def custom_embedding_reward(prompts, completions, dataset=None, sbert=None, **kwargs):
    """
    For each example:
    - Check if the final answer matches exactly (reward 1 or 0).
    - If correct, compute mean max sentence similarity reward on reasoning traces.
    - Final reward = correctness * embedding similarity.
    
    Returns:
        List of float rewards in [0,1].
    """

    if dataset is None:
        raise ValueError("'dataset' must be provided.")

    rewards = []

    for i, completion in enumerate(completions):
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get("content", "")
        elif isinstance(completion, dict):
            text = completion.get("content", "")
        else:
            text = str(completion)

        # Extract gold answer and reasoning text from dataset
        gold_answer_and_reasoning = dataset[i].get("answer", "")
        gold_reasoning = extract_xml_think(gold_answer_and_reasoning)
        gold_answer = standardize_label((extract_xml_answer(gold_answer_and_reasoning)))


        # Extract generated answer and reasoning from text (generated completion)
        generated_reasoning = extract_xml_think(text)
        generated_answer = standardize_label(extract_xml_answer(text))

        # BinaryCorrectness reward: 1 if exact match, else 0 (this will get multiplied later)
        correctness = 1.0 if generated_answer == gold_answer else 0.0

        # If answer incorrect, skip embedding similarity computation, return reward = 0 
        if correctness == 0.0:
            rewards.append(0.0)
            continue

        # If generated or gold reasoning text missing, reward 0
        if not generated_reasoning or not gold_reasoning:
            rewards.append(0.0)
            continue

        # Split into sentences to compute embeddings using . as a delimiter
        gold_sentences = [s.strip() for s in gold_reasoning.split('.') if s.strip()]
        gen_sentences = [s.strip() for s in generated_reasoning.split('.') if s.strip()]
        

        # Compute embeddings
        gold_embeddings = sbert.encode(gold_sentences, convert_to_tensor=True, normalize_embeddings=True)
        gen_embeddings = sbert.encode(gen_sentences, convert_to_tensor=True, normalize_embeddings=True)

        # Cosine similarity matrix
        sim_matrix = util.pytorch_cos_sim(gold_embeddings, gen_embeddings)

        # Mean max similarity per gold sentence, scaled [0,1]
        max_sim_per_gold = sim_matrix.max(dim=1).values
        embedding_reward = ((max_sim_per_gold.mean().item()) + 1) / 2

        # Final reward = correctness * embedding similarity (correctness=1 here already)
        rewards.append(embedding_reward)

    return rewards
