# 🧠 Reasoning Agent for ACR-Based Decision Justification

This folder contains all components related to the training, evaluation, and extraction of structured reasoning traces in medical imaging, aligned with ACR Appropriateness Criteria.

---

## `reasoning_agent.py`

### 🔍 Overview
Main class for LoRA-based instruction tuning and inference on structured reasoning traces using ACR variant, procedure, and citations.

### 🔧 Key Methods
- `prepare_dataset(json_path)`  
  Loads the ACR dataset with citations and generates structured prompts using XML format: `<think>...</think><answer>...</answer>`.
- `run_inference(...)`  
  Loads LoRA adapters and runs batched inference on the test set, extracting both reasoning and answer components.
  You can download the Lora Adaptors from my HuggingFace repo `huggingface.co/annietz/grpo-acr-adapters`
---

## `train_rl_reasoning_agent.py`

### 🔍 Overview
Trains the Reasoning Agent with GRPO (Group Relative Policy Optimization) using one or more custom reward functions.

### 🔧 Key Components
- Supports reward types:
  - `format` (XML structure)
  - `answer` (correctness of label)
  - `llm_eval` (LLM-judged reasoning quality)
  - `custom_embedding` (reasoning trace alignment via SBERT)
- Uses `GRPOTrainer` from TRL to apply multiple reward functions jointly.

---

## `reward_functions.py`

### 🎯 Contains Reward Functions:
- `format_reward` – Enforces output structure with XML tags.
- `correctness_answer` – Binary reward for correct answer label.
- `llm_eval_reward` – Scores generated reasoning vs. gold using LLM on clinical alignment.
- `custom_embedding_reward` – Measures cosine similarity between reasoning sentences (SBERT).

---

## `train_sft_model.py`

### 🔍 Overview
Performs supervised fine-tuning (SFT) using standard ACR labels and simplified prompts (no reasoning).

### 🧠 Prompt Format
System prompt + clinical input, output is `<answer>...</answer>`.

---

## `model_evaluation.py`

### 📊 Evaluates final model performance:
- `evaluate_classification()` – F1 & confusion matrix for final answer.
- `evaluate_entity_embedding_alignment()` – Precision, recall, F1 on extracted clinical entities.
- `evaluate_llm_alignment()` – 0–10 alignment score via LLM comparison with gold reasoning.

You would need to run inference on using the Agent and parse the results before running the evaluation. An example of the format can be found in `data/example_results.json`.

See the arguments to understand how to run evaluation. An example can be:

```bash
python model_evaluation.py --path data/generalization_dataset_example.json --classification --entities --llm
```

---

## `reward_plotting.py`

### 📈 Visualizes training rewards:
- Smooths and plots reward logs over epochs for each reward type (answer, format, etc.). You need the `trainer_state` files for this.

---

## `reasoning_extraction.py`

### 🧠 Reasoning Trace Extraction
Extracts step-by-step reasoning traces from full ACR justifications using LLaMA 4 Scout.
Used to generate gold-standard reasoning traces for reward evaluation.

---

## `pdf_extractor.py`

### 🔍 PDF extraction
PDF extraction from the ACR Narrative documents by identifying procedure sections, and preparing extracted content for downstream processing. Some manual work is still needed due to format issues.



---

## `data/example_results.json`

### 🧠 Examples of reasoning
Random examples from the MedReason-Embed model showing two responses — one correct and one incorrect — to illustrate how the reasoning works.

