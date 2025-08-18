import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# import all your file paths that contain the final results for the models you want to run the significance tests on
file_paths = [
    "",
    ""
    ]

model_names = [
    "",
    ""
]

def normalize_answer(ans):
    if not isinstance(ans, str):
        return ""
    return ans.strip().lower().replace('(', '').replace(')', '')

# Load gold labels from first file
with open(file_paths[0], "r") as f:
    data = json.load(f)
gold_labels = [normalize_answer(item.get('gold_answer', '')) for item in data]
n_samples = len(gold_labels)

correct_matrix = []
for path in file_paths:
    with open(path, "r") as f:
        data = json.load(f)
    model_preds = [normalize_answer(item.get('llm_answer', '')) for item in data]
    
    if len(model_preds) != n_samples:
        raise ValueError(f"File {path} length {len(model_preds)} does not match gold label count {n_samples}")
    
    correct = [int(pred == gold) for pred, gold in zip(model_preds, gold_labels)]
    correct_matrix.append(correct)

correct_matrix = np.array(correct_matrix).T  # Now shape (n_samples, n_models)

print("\nPairwise McNemar's test p-values (lower means more significant difference):\n")
n_models = len(file_paths)
for i in range(n_models):
    for j in range(i + 1, n_models):
        print(f"Comparing {model_names[i]} vs {model_names[j]} ...")
        model_i = correct_matrix[:, i]
        model_j = correct_matrix[:, j]

        both_correct = np.sum((model_i == 1) & (model_j == 1))
        only_i = np.sum((model_i == 1) & (model_j == 0))
        only_j = np.sum((model_i == 0) & (model_j == 1))
        both_wrong = np.sum((model_i == 0) & (model_j == 0))
        table = [[both_correct, only_i], [only_j, both_wrong]]

        try:
            result = mcnemar(table, exact=True)
            sig = "SIGNIFICANT" if result.pvalue < 0.01 else "not significant"
            print(f"{model_names[i]} vs. {model_names[j]}: p = {result.pvalue:.4f}  --> {sig}")
        except Exception as e:
            print(f"Error running McNemar test for {model_names[i]} vs {model_names[j]}: {e}")