import pandas as pd
import numpy as np
import os
import json

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon
from statsmodels.sandbox.stats.multicomp import multipletests


import matplotlib.pyplot as plt
import seaborn as sns

path = "/content/drive/MyDrive/Research2025/Semeval2024/ClasificadoresNew/ablation/"

df_ablation = pd.read_csv(path + "ablation_results.csv")
df_ablation

df_ablation_sorted = df_ablation.sort_values(by='F1', ascending=False)
df_ablation_sorted

output_dir = path + "statistical_significance"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(12, 8))
sns.barplot(x='Variant', y='F1', hue='Classifier', data=df_ablation)
plt.title('F1 Score by Variant and Classifier')
plt.xlabel('Variant')
plt.ylabel('F1 Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'f1_score_barplot.png')) # Save the plot
plt.show()

top_6_models = df_ablation_sorted.head(6)

probability_files = []

for index, row in top_6_models.iterrows():
    variant = row['Variant']
    classifier = row['Classifier'].replace(" ", "").replace("(", "").replace(")", "")

    # Construct the filename based on the provided examples and corrected pattern
    if "LOO - no" in variant:
        variant_filename = variant.replace("LOO - no ", "LOO_-_no_")
        filename = f"{variant_filename}_{classifier}_predictions.jsonl"
    elif "Single -" in variant:
        variant_filename = variant.replace("Single - ", "Single_-_")
        filename = f"{variant_filename}_{classifier}_predictions.jsonl"
    else:
        variant_filename = variant.replace(" ", "_")
        filename = f"{variant_filename}_{classifier}_predictions.jsonl"

    probability_files.append(filename)


print("\nProbability Files:")
print(probability_files)

true_labels = {}
probabilities = {}

predictions_path = path + "predictions/"

# Use the probability_files list generated in the previous step
for filename in probability_files:
    filepath = predictions_path + filename

    labels = []
    probs = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                prediction_data = json.loads(line)
                # Assuming the keys are 'label' for true labels and 'prediction_probability' for probabilities
                labels.append(prediction_data['label'])
                probs.append(prediction_data['prediction_probability'])
        # Extract the model name from the filename (remove "_predictions.jsonl")
        model_name = filename.replace("_predictions.jsonl", "")
        true_labels[model_name] = np.array(labels)
        probabilities[model_name] = np.array(probs)
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except KeyError as e:
        print(f"KeyError: {e} in file: {filepath}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {filepath}")


print("Keys in true_labels dictionary:")
print(true_labels.keys())
print("\nKeys in probabilities dictionary:")
print(probabilities.keys())


file_list = os.listdir(path)
print(file_list)

# Function to perform Wilcoxon signed-rank test
def perform_wilcoxon_test(model1_probs, model2_probs, true_labels):
    """
    Performs Wilcoxon signed-rank test to compare two classifiers based on probabilities.

    Args:
        model1_probs (np.array): Predicted probabilities for model 1.
        model2_probs (np.array): Predicted probabilities for model 2.
        true_labels (np.array): True labels.

    Returns:
        tuple: Wilcoxon statistic and p-value.
    """
    # Calculate the difference in probabilities for the positive class (assuming it's class 1)
    # We need to compare the performance, so we look at the difference in errors or correct predictions.
    # A common approach is to compare the scores (probabilities for the true class)
    # For class 1, the score is the probability of class 1.
    # For class 0, the score is 1 - probability of class 1.
    model1_scores = np.where(true_labels == 1, model1_probs, 1 - model1_probs)
    model2_scores = np.where(true_labels == 1, model2_probs, 1 - model2_probs)

    # Calculate the differences in scores
    score_differences = model1_scores - model2_scores

    # Perform Wilcoxon signed-rank test on the differences
    # We use 'zero_method='correction'' to account for tied differences at zero
    # We use 'correction=True' for continuity correction
    statistic, pvalue = wilcoxon(score_differences, zero_method='pratt', correction=False)


    return statistic, pvalue


# Get the list of model names
model_names = list(true_labels.keys())

# Perform McNemar's and Wilcoxon tests for each pair of models
statistical_results = {}

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        model1_name = model_names[i]
        model2_name = model_names[j]

        # Convert probabilities to predicted labels (assuming a threshold of 0.5 for McNemar)
        model1_pred_labels = (probabilities[model1_name] > 0.5).astype(int)
        model2_pred_labels = (probabilities[model2_name] > 0.5).astype(int)

        # Assuming all models have the same true labels
        true_labels_array = list(true_labels.values())[0]

        # Perform Wilcoxon signed-rank test
        wilcoxon_statistic, wilcoxon_pvalue = perform_wilcoxon_test(
            probabilities[model1_name],
            probabilities[model2_name],
            true_labels_array
        )


        statistical_results[f"{model1_name} vs {model2_name}"] = {
            'wilcoxon_statistic': wilcoxon_statistic,
            'wilcoxon_pvalue': wilcoxon_pvalue
        }

print("Statistical Test Results:")
for pair, results in statistical_results.items():
    print(f"{pair}:")
    print(f"  Wilcoxon Test: Statistic = {results['wilcoxon_statistic']}, P-value = {results['wilcoxon_pvalue']}")

statistical_results

# Create a list to hold the results for the Wilcoxon DataFrame
wilcoxon_results_list = []


# Extract results from the statistical_results dictionary
for pair, results in statistical_results.items():
    model1_name, model2_name = pair.split(" vs ")

    wilcoxon_results_list.append({
        'Model_A': model1_name,
        'Model_B': model2_name,
        'Wilcoxon_stat': results['wilcoxon_statistic'],
        'p_value': results['wilcoxon_pvalue']
    })


# Create DataFrames from the lists
wilcoxon_results_df = pd.DataFrame(wilcoxon_results_list)

# --- Add print statements here to inspect the DataFrames before saving and styling ---
print("Wilcoxon Results DataFrame before styling/saving:")
print(wilcoxon_results_df[['Model_A', 'Model_B', 'p_value', 'Wilcoxon_stat']])
# ----------------------------------------------------------------------------------


# Perform multiple testing correction for Wilcoxon results
reject_wilcoxon, pvals_corrected_wilcoxon, _, _ = multipletests(wilcoxon_results_df['p_value'], method='holm')
wilcoxon_results_df['p_value_corrected'] = pvals_corrected_wilcoxon
wilcoxon_results_df['Significant'] = reject_wilcoxon


# Display the results in the requested format with more decimal places
print("\nWilcoxon Signed-Rank Test Results (Styled Display):")
display(wilcoxon_results_df[['Model_A', 'Model_B', 'Wilcoxon_stat', 'p_value', 'p_value_corrected', 'Significant']].style.format({'p_value': '{:.15f}', 'p_value_corrected': '{:.15f}'}))


# Save Wilcoxon results
#wilcoxon_csv_path = os.path.join(output_dir, "wilcoxon_results.csv")
#wilcoxon_jsonl_path = os.path.join(output_dir, "wilcoxon_results.jsonl")

#wilcoxon_results_df.to_csv(wilcoxon_csv_path, index=False)
#wilcoxon_results_df.to_json(wilcoxon_jsonl_path, orient='records', lines=True)

#print(f"Wilcoxon results saved to: {wilcoxon_csv_path} and {wilcoxon_jsonl_path}")

# Summarize the findings
print("Summary of Statistical Significance Test Results (alpha = 0.05):")
wilcoxon_significant_differences = []

alpha = 0.05 # Define alpha here

for pair, results in statistical_results.items():
    if results['wilcoxon_pvalue'] < alpha:
        wilcoxon_significant_differences.append(pair)

print("\nStatistically significant differences based on Wilcoxon Test:")
if wilcoxon_significant_differences:
    for pair in wilcoxon_significant_differences:
        print(f"- {pair}")
else:
    print("No statistically significant differences found based on Wilcoxon Test.")

# Prepare data for heatmap
model_names = list(true_labels.keys()) # Get model names in a consistent order
n_models = len(model_names)

# Create a matrix to store significance results
# Initialize with False (not significant)
significance_matrix_wilcoxon = np.full((n_models, n_models), False)

# Map model names to indices
name_to_index = {name: i for i, name in enumerate(model_names)}

# Populate the significance matrix
for index, row in wilcoxon_results_df.iterrows():
    model_a = row['Model_A']
    model_b = row['Model_B']
    is_significant = row['Significant']

    # Get indices
    idx_a = name_to_index[model_a]
    idx_b = name_to_index[model_b]

    # Fill the matrix for both pairs (A vs B and B vs A)
    significance_matrix_wilcoxon[idx_a, idx_b] = is_significant
    significance_matrix_wilcoxon[idx_b, idx_a] = is_significant

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(significance_matrix_wilcoxon, annot=True, cmap='viridis', fmt='d',
            xticklabels=model_names, yticklabels=model_names, cbar=False)
plt.title('Statistically Significant Differences (Wilcoxon Test)')
plt.xlabel('Model B')
plt.ylabel('Model A')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'wilcoxon_heatmap.png')) # Save the plot
plt.show()

!pip freeze > requirements.txt