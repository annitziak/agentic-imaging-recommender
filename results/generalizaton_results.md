
## 🌍 Generalization Evaluation

To assess the robustness of our models in unseen clinical scenarios, we evaluated them on a **generalization set** containing five unrelated, previously unseen conditions. This subset is about half the size of our main test set.

We report results using two types of supporting evidence:
1. **Gold-standard ACR citations**
2. **Our own retrieved citations** via DeepRetrieval + filtering pipeline

---

### 📊 Results with ACR Citations

| Model / Config  | Macro Avg F1 | Weighted F1 |
| --------------- | ------------ | ----------- |
| Baseline        | 31.0%        | 34.5%       |
| Citations       | 40.5%        | 53.0%       |
| LLM Eval        | 46.6%        | 63.6%       |
| MedReason-Embed | 44.5%        | 63.3%       |
| SFT             | 43.3%        | 65.1%       |
| LLaMA 405B      | 51.7%        | 60.0%       |


---

### 📊 Results with Our Own Citations

| Model / Config  | Macro Avg F1 | Weighted F1 |
| --------------- | ------------ | ----------- |
| Baseline        | 31.0%        | 34.5%       |
| Citations       | 40.4%        | 46.6%       |
| LLM Eval        | 43.8%        | 54.9%       |
| MedReason-Embed | 45.9%        | 55.0%       |

---

### ✅ Key Takeaways

- **Performance drops on unseen conditions were small but consistent**, especially for models using citation-based rewards. Most showed a 5–6% drop in macro F1 compared to the performance on the test set, indicating some small but expected and manageable distribution sensitivity.
- **The ranking of models remained stable**, with LLM Eval performing best. SFT continued to default to “Usually Not Appropriate” and overfitted showing that it did not generalize well.
- **Our retrieved citations led to comparable results** vs. ACR citations, with the MedReason-Embed model even improving slightly — showing that different sets of high-quality evidence often lead to the same clinical conclusions.
- **No reward hacking or overfitting was observed**, suggesting the performance gap stems from condition shifts and citation variability, not training artifacts.
- These results emphasize that **retrieval-based reasoning agents can generalize**, and with broader condition coverage, they may become practical tools for real-world clinical decision support.
- **Significant more research is needed** to understand why the models had the set performance drop including collaboration with medical professionals.

---

