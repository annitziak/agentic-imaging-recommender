# 🧠 ICD Coding & ACR Variant Matching

This folder contains two key modules in the medical reasoning system. The first standardizes noisy clinical inputs and maps them to ICD-9-CM codes. The second matches ICD-labeled inputs to the most relevant ACR variant.

---

## `ICD_Coding_Agent.py`

### 🔍 Overview
Maps unstructured or noisy diagnostic notes to clean ICD-9-CM codes using a pipeline of LLM-based normalization, embedding-based retrieval, and optional LLM reranking.

### 🔧 Main Pipeline
1. **Normalization** (`normalize_clinical_term`)  
   - LLaMA 3 rewrites the input to a standardized English medical term using a clinical prompt.
   - Ignores gender, age, or location for alignment with ICD standard.

2. **Semantic Retrieval** (`search`)  
   - SentenceTransformer (e.g., BioBERT) encodes the term.
   - FAISS retrieves top-K most similar ICD codes from a prebuilt index.

3. **LLM Reranking** (`llm_rerank`)  
   - Triggered if score distribution is low-confidence. This is set artificially low so the LLM also functions as a "second check".
   - Reranks retrieved ICDs using LLaMA based on semantic similarity to the normalized query.

4. **Evaluation** (`evaluate`)  
   - Supports `Top-1 Accuracy`, `Top-5 Accuracy`, `Hierarchical Top-1 Accuracy`, and `MRR`.
   - Works best with a dataframe structure (new column with top k predicted codes) and the ground truth needs to be available.

### ▶️ Example
```python
agent = ICD_Coding_Agent(
    faiss_index_path="data/icd_index.faiss",
    codes_path="data/icd_codes.json"
)
term = agent.normalize_clinical_term("tumori maligni della prostata")
results = agent.search(term)
```

---

## `ACR_Criteria_Checker.py`

### 🔍 Overview
Maps a given ICD code and clinical note to the most relevant **ACR variant** (e.g., *"Variant 3: Female age 30–39..."*) using rule-based filtering and semantic similarity.

### Matching Logic

- **If the ICD code is found** in our predefined ACR condition list  
  (mapped offline using the same `ICD_Coding_Agent` pipeline and stored in `ACR_ICD_Mapping.xlsx`. This is just a sample for `Breast Pain`).
  1. Apply ICD code filtering.
  2. Match based on patient **age** and **anamnesis**.
  3. If no exact match:
     - Fall back to **sentence similarity**.
     - If still no good match, default to **Variant 1**.

- **If the ICD code is not found** in the ACR list:  
  - Trigger the `Medical_Review_Agent` to retrieve and process relevant evidence.

---

### ▶️ Example
```python
df = pd.read_excel("data/ACR_ICD_Mapping.xlsx")
checker = ACRCriteriaChecker(df)
variant = checker.matching_variant("Adult with chest pain. F39", "61171")
```

---