# 📚 Literature Retrieval and Post-Filtering Modules

This directory contains all components related to retrieving medical literature (from PubMed and ACR), extracting high-quality evidence, and evaluating retrieval quality.

---

## `Medical_Review_Agent.py`

### 🔍 Overview
Orchestrates the full pipeline of literature retrieval for a clinical condition or imaging procedure.

### 🔧 Main Steps
1. **Prompt Generation** (`make_prompts_for_deepretrieval`) – Generates prompts for DeepRetrieval and PubMed.
   - DeepRetrieval must be run separately (e.g., using `vllm_host.sh`) and output stored in JSON format with keys `"condition"` and `"rewritten_queries"`. For more information see : `github.com/pat-jj/DeepRetrieval`.
   - Example can be found in: `data/deepretrieval_queries_generalization_set.json`
2. **Abstract Retrieval** (`process_variants`) – Retrieves abstracts from PubMed for each query.
3. **Evidence Strength Prediction** (`predict_strength_of_evidence`) – Uses `PostFilteringAgent` to assign study quality scores.
4. **Filtering and Condensing** (`filter_high_quality_abstracts`, `condense_abstracts`) – Keeps only high-quality studies and condesnes abstracts into key sections (e.g., results, conclusions).
5. **Final Redirection** (`redirect_in_variants`) – Redirects the filtered abstracts into condition-variant-procedure pairs by applying simple heuristics. 

---

## `Post_Filtering_Agent.py`

### 🔍 Overview
Trains and uses a Random Forest model to predict the strength of evidence from abstract content and journal metadata, following GRADE-style logic.

### 🧠 Core Logic
- **Feature Extraction** (`_extract_grade_features`) – Identifies features like presence of diagnostic accuracy terms, outcome metrics, comparators, RCTs, confidence intervals, and more.
- **Journal Matching** (`_add_sjr_feature`) – Adds SJR scores to entries using exact and fuzzy matching of journal names.
- **Model Training** (`train_model`) – Trains a Random Forest using extracted features and publication metadata.
- **Prediction** (`predict_quality`) – Predicts evidence strength for unseen papers using saved models.

---

## `literature_scraper_acr.py`

### 🔍 Overview
Extracts ACR guideline variants and their associated citations from Appendix PDFs and enriches them with PubMed abstracts.

### 🧰 Key Functions
- **Variant Parsing** (`parse_pdf_variants`) – Extracts "Variant X" sections and their cited PMIDs from ACR PDFs.
- **Abstract Fetching** (`fetch_pubmed_abstract`) – Queries PubMed API using extracted PMIDs.
- **Merging & Cleaning** (`merge_all_outputs`, `condense_and_filter_json_file`) – Combines all extracted entries, extracts only key abstract sections (results/conclusions), and filters for word limit.

---

## `eval_retrieval_strategy.py`

### 🔍 Overview
Evaluates whether retrieved literature (e.g., from DeepRetrieval or PubMed) covers similar semantic ground as gold ACR citations.

- **Embedding & Similarity** (`embed_abstracts`, `average_min_center_distance`) – Encodes and compares abstract embeddings (BioBERT).
- **Clustering & Visualization** (`cluster_and_get_distribution`, `plot_embeddings_with_clusters`) – Clusters abstracts with KMeans and visualizes with PCA to compare distributions.
- **Topic Modeling** (`run_lda`) – Runs LDA to compare topic diversity and overlap between gold vs. retrieved abstracts.

---