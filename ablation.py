# -*- coding: utf-8 -*-


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import gc

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
tqdm.pandas()
import random
import joblib
import os

import ast
from typing import Dict, Tuple, List

import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

RANDOM_STATE = 42


import warnings
warnings.filterwarnings("ignore")


path_ft = "/content/drive/MyDrive/Research2025/Raid/dfs_Preproc_Caract/"

df_train = pd.read_json(path_ft + 'df_train_combined_OK_New.jsonl', orient='records', lines=True)
df_test = pd.read_json(path_ft + 'df_test_combined_OK_New.jsonl', orient='records', lines=True)

df_train

print("Shape of df_train:", df_train.shape)
print("Shape of df_test:", df_test.shape)

print(df_train["label"].value_counts(), flush=True)


# Define the directory to save models and results
model_dir = "/content/drive/MyDrive/Research2025/Raid/ClasificadoresNew/ablation/"
os.makedirs(model_dir, exist_ok=True)
results_dir = os.path.join(model_dir, "predictions")
os.makedirs(results_dir, exist_ok=True)


# ============================================
# 1) Utilities: parsing and stacking of vectorial columns
# ============================================
def parse_array_cell(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float32)
    if isinstance(x, (int, float)):
        return np.array([x], dtype=np.float32)
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return np.array(v, dtype=np.float32)
            elif isinstance(v, (int, float)):
                return np.array([float(v)], dtype=np.float32)
        except Exception:
            try:
                return np.array([float(x)], dtype=np.float32)
            except Exception:
                pass
    raise ValueError(f"The cell could not be parsed to vector: {x}")

def stack_block(series) -> np.ndarray:
    mats = [parse_array_cell(v) for v in series.tolist()]
    dims = [m.shape[0] for m in mats]
    if len(set(dims)) != 1:
        # Attempt to handle inconsistent dimensions by padding with NaNs
        max_dim = max(dims)
        padded_mats = []
        for m in mats:
            if m.shape[0] < max_dim:
                padded_mats.append(np.pad(m, (0, max_dim - m.shape[0]), 'constant', constant_values=np.nan))
            else:
                padded_mats.append(m)
        return np.stack(padded_mats, axis=0).astype(np.float32)

    return np.stack(mats, axis=0).astype(np.float32)


# ============================================
# 2) Preparation of blocks (Style, Feat, Pplx, TFIDF+SVD)
# ============================================
from sklearn.impute import SimpleImputer

def build_blocks(df_train: pd.DataFrame, df_test: pd.DataFrame,
                 tfidf_max_features=50000, svd_components=300) -> Dict[str, Dict[str, np.ndarray]]:
    print("Building blocks...", flush=True)
    # Numerical blocks already present
    style_tr = stack_block(df_train["style_distance_embedding"]); style_te = stack_block(df_test["style_distance_embedding"])
    feat_tr  = stack_block(df_train["feature_select"]);  feat_te  = stack_block(df_test["feature_select"])
    pplx_tr  = stack_block(df_train["perplexity"]); pplx_te = stack_block(df_test["perplexity"])

    # TF-IDF + SVD from processed_text
    print("Applying TF-IDF and SVD...", flush=True)
    tfidf = TfidfVectorizer(
        max_features=tfidf_max_features,
        ngram_range=(1,4), # Updated ngram_range
        lowercase=True, strip_accents='unicode',
        min_df=2
    )
    Xtr_tfidf = tfidf.fit_transform(tqdm(df_train["processed_text"], desc="TF-IDF Fit Transform (Train)"))
    Xte_tfidf = tfidf.transform(tqdm(df_test["processed_text"], desc="TF-IDF Transform (Test)"))

    gc.collect() # Free memory after creating large TF-IDF matrices

    svd = TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)
    tfidf_tr = svd.fit_transform(Xtr_tfidf).astype(np.float32)
    tfidf_te = svd.transform(Xte_tfidf).astype(np.float32)

    del Xtr_tfidf, Xte_tfidf # Delete intermediate TF-IDF matrices
    gc.collect() # Free memory again

    # Scaling per block (fit on train)
    print("Scaling blocks...", flush=True)
    sc_style = StandardScaler().fit(style_tr)
    sc_feat  = StandardScaler().fit(feat_tr)
    sc_pplx  = StandardScaler().fit(pplx_tr)
    sc_tfidf = StandardScaler().fit(tfidf_tr)

    style_tr_scaled = sc_style.transform(style_tr)
    style_te_scaled = sc_style.transform(style_te)
    feat_tr_scaled = sc_feat.transform(feat_tr)
    feat_te_scaled = sc_feat.transform(feat_te)
    pplx_tr_scaled = sc_pplx.transform(pplx_tr)
    pplx_te_scaled = sc_pplx.transform(pplx_te)
    tfidf_tr_scaled = sc_tfidf.transform(tfidf_tr)
    tfidf_te_scaled = sc_tfidf.transform(tfidf_te)

    del style_tr, style_te, feat_tr, feat_te, pplx_tr, pplx_te, tfidf_tr, tfidf_te # Delete unscaled blocks
    gc.collect() # Free memory again

    # Impute potential NaNs after scaling
    print("Imputing NaNs...", flush=True)
    imputer = SimpleImputer(strategy='mean')

    style_tr_imputed = imputer.fit_transform(style_tr_scaled)
    style_te_imputed = imputer.transform(style_te_scaled)
    feat_tr_imputed = imputer.fit_transform(feat_tr_scaled)
    feat_te_imputed = imputer.transform(feat_te_scaled)
    pplx_tr_imputed = imputer.fit_transform(pplx_tr_scaled)
    pplx_te_imputed = imputer.transform(pplx_te_scaled)
    tfidf_tr_imputed = imputer.fit_transform(tfidf_tr_scaled)
    tfidf_te_imputed = imputer.transform(tfidf_te_scaled)

    del style_tr_scaled, style_te_scaled, feat_tr_scaled, feat_te_scaled, pplx_tr_scaled, pplx_te_scaled, tfidf_tr_scaled, tfidf_te_scaled # Delete scaled blocks
    gc.collect() # Free memory again


    blocks = {
        "style": {"train": style_tr_imputed, "test": style_te_imputed},
        "feat":  {"train": feat_tr_imputed,   "test": feat_te_imputed},
        "pplx":  {"train": pplx_tr_imputed,   "test": pplx_te_imputed},
        "tfidf": {"train": tfidf_tr_imputed, "test": tfidf_te_imputed},
    }

    transformers = {
        "tfidf": tfidf,
        "svd": svd,
        "sc_style": sc_style,
        "sc_feat": sc_feat,
        "sc_pplx": sc_pplx,
        "sc_tfidf": sc_tfidf,
        "imputer": imputer
    }


    print("Blocks built successfully.", flush=True)
    return blocks, transformers


def concat_blocks(blocks: Dict[str, Dict[str, np.ndarray]], use_flags: Dict[str, bool]) -> Tuple[np.ndarray, np.ndarray]:
    parts_tr, parts_te = [], []
    for key in ["style", "feat", "pplx", "tfidf"]:
        if use_flags.get(key, False):
            parts_tr.append(blocks[key]["train"])
            parts_te.append(blocks[key]["test"])
    if not parts_tr: # Handle case where no blocks are selected
      return None, None
    X_train = np.concatenate(parts_tr, axis=1) if len(parts_tr) > 1 else parts_tr[0]
    X_test  = np.concatenate(parts_te, axis=1) if len(parts_te) > 1 else parts_te[0]
    return X_train, X_test


# ============================================
# 3) Train/Evaluate a classifier
# ============================================
from sklearn.svm import LinearSVC # Changed from SVC to LinearSVC

def eval_classifier(clf, Xtr, ytr, Xte, yte):
    # Training time
    start_time = time.time()
    clf.fit(Xtr, ytr)
    train_time = time.time() - start_time

    # Inference time
    start_time = time.time()
    pred = clf.predict(Xte)
    inference_time = time.time() - start_time

    # Inference speed (samples per second)
    inference_speed = len(yte) / inference_time if inference_time > 0 else 0

    acc = accuracy_score(yte, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)

    # Classification report and confusion matrix
    class_report = classification_report(yte, pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(yte, pred).tolist() # Convert to list for easier storage

    # Get prediction probabilities
    if hasattr(clf, "predict_proba"):
        probabilities = clf.predict_proba(Xte)
        probability_class_0 = probabilities[:, 0]
        probability_class_1 = probabilities[:, 1]
        # For binary classification, prediction probability can be probability of class 1
        prediction_probability = probability_class_1
    elif hasattr(clf, "decision_function"):
        # For models that don't have predict_proba but have decision_function (like LinearSVC)
        decision_values = clf.decision_function(Xte)
        # For binary classification, decision_function returns scores. We can scale these to
        # pseudo-probabilities, but for simplicity, we'll just use the decision value as
        # a proxy for confidence for now, or you can apply a sigmoid.
        # A common approach is to use sigmoid for pseudo-probabilities: 1 / (1 + exp(-decision_values))
        # Let's use the sigmoid approach for probability_class_1
        probability_class_1 = 1 / (1 + np.exp(-decision_values))
        probability_class_0 = 1 - probability_class_1
        prediction_probability = probability_class_1 # Or you might choose the absolute decision value
    else:
        # Handle cases where neither predict_proba nor decision_function is available
        probabilities = np.full((len(yte), 2), np.nan)
        probability_class_0 = np.full(len(yte), np.nan)
        probability_class_1 = np.full(len(yte), np.nan)
        prediction_probability = np.full(len(yte), np.nan)


    return acc, prec, rec, f1, train_time, inference_time, inference_speed, class_report, conf_matrix, pred, prediction_probability, probability_class_0, probability_class_1


# ============================================
# 4) Ablation and classifier configurations
# ============================================
ABLATIONS = {
    "Full model":      {"style": True,  "feat": True,  "pplx": True,  "tfidf": True},
    "LOO - no Style":  {"style": False, "feat": True,  "pplx": True,  "tfidf": True},
    "LOO - no Feat":   {"style": True,  "feat": False, "pplx": True,  "tfidf": True},
    "LOO - no Pplx":   {"style": True,  "feat": True,  "pplx": False, "tfidf": True},
    "LOO - no TFIDF":  {"style": True,  "feat": True,  "pplx": True,  "tfidf": False},
    "Single - Style":  {"style": True,  "feat": False, "pplx": False, "tfidf": False},
    "Single - Feat":   {"style": False, "feat": True,  "pplx": False,  "tfidf": False}, # Corrected flags for Single - Feat
    "Single - Pplx":   {"style": False, "feat": False, "pplx": True,  "tfidf": False}, # Corrected flags for Single - Pplx
    "Single - TFIDF":  {"style": False, "feat": False, "pplx": False, "tfidf": True}, # Corrected flags for Single - TFIDF
}

def get_classifiers():
    # Note: you can activate class_weight="balanced" if you have imbalance.
    clfs = {
        "SVM(LinearSVC)": LinearSVC(C=1.0, random_state=RANDOM_STATE),
        "RandomForest":   RandomForestClassifier(
            n_estimators=500, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "LogisticReg":    LogisticRegression(
            C=1.0, solver="saga", max_iter=2000, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    return clfs

# ============================================
# 5) Run everything
# ============================================
# y_train / y_test
y_train = df_train["label"].astype(int).values
y_test  = df_test["label"].astype(int).values

# Build scaled blocks
blocks, transformers = build_blocks(df_train, df_test, tfidf_max_features=3000, svd_components=300) # Updated tfidf_max_features

# Save transformers
print("\nSaving transformers...", flush=True)
for name, transformer in tqdm(transformers.items(), desc="Saving transformers"):
    joblib.dump(transformer, os.path.join(model_dir, f"{name}.pkl"))
print("Transformers saved.", flush=True)

# Ablation and classifier loop
rows = []
print("\nStarting ablation and classifier evaluation...", flush=True)
for abl_name, flags in tqdm(ABLATIONS.items(), desc="Ablations"):
    Xtr, Xte = concat_blocks(blocks, flags)
    assert Xtr is not None, f"{abl_name}: no hay bloques activos"
    clfs = get_classifiers()
    for clf_name, clf in tqdm(clfs.items(), desc=f"Classifiers for {abl_name}", leave=False):
        acc, prec, rec, f1, train_time, inference_time, inference_speed, class_report, conf_matrix, pred, prediction_probability, probability_class_0, probability_class_1 = eval_classifier(clf, Xtr, y_train, Xte, y_test)
        rows.append({
            "Variant": abl_name, "Classifier": clf_name,
            "Accuracy": f"{acc:.6f}", "Precision": f"{prec:.6f}",
            "Recall": f"{rec:.6f}", "F1": f"{f1:.6f}",
            "Train Time (s)": round(train_time, 4),
            "Inference Time (s)": round(inference_time, 4),
            "Inference Speed (samples/s)": round(inference_speed, 2),
            "Classification Report": class_report,
            "Confusion Matrix": conf_matrix
        })
        print(f"[TEST] {abl_name} | {clf_name} -> ACC {acc:.6f} | P {prec:.6f} | R {rec:.6f} | F1 {f1:.6f} | Train Time {train_time:.4f}s | Inference Time {inference_time:.4f}s | Inference Speed {inference_speed:.2f} samples/s", flush=True)

        # Print classification report
        print(f"\nClassification Report for {abl_name} - {clf_name}:", flush=True)
        print(classification_report(y_test, pred, zero_division=0, digits=6), flush=True)


        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {abl_name} - {clf_name}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        # Save confusion matrix plot
        confusion_matrix_filename = os.path.join(model_dir, f"{abl_name.replace(' ', '_')}_{clf_name.replace('(', '').replace(')', '')}_confusion_matrix.png")
        plt.savefig(confusion_matrix_filename)
        plt.show()
        plt.close() # Close the plot to free memory


        # Save the trained model
        model_filename = os.path.join(model_dir, f"{abl_name.replace(' ', '_')}_{clf_name.replace('(', '').replace(')', '')}.pkl")
        joblib.dump(clf, model_filename)
        print(f"Saved model to {model_filename}", flush=True)

        # Add predictions to df_test and save
        df_test_results = df_test.copy()
        df_test_results["predicted_label"] = pred
        df_test_results["prediction_probability"] = prediction_probability
        df_test_results["probability_class_0"] = probability_class_0
        df_test_results["probability_class_1"] = probability_class_1

        results_filename = os.path.join(results_dir, f"{abl_name.replace(' ', '_')}_{clf_name.replace('(', '').replace(')', '')}_predictions.jsonl")
        df_test_results.to_json(results_filename, orient='records', lines=True)
        print(f"Saved predictions for {abl_name} - {clf_name} to {results_filename}", flush=True)


results_df = pd.DataFrame(rows)
# Output order
order = ["Full model",
         "LOO - no Style", "LOO - no Feat", "LOO - no Pplx", "LOO - no TFIDF",
         "Single - Style", "Single - Feat", "Single - Pplx", "Single - TFIDF"]
results_df["Variant"] = pd.Categorical(results_df["Variant"], categories=order, ordered=True)
results_df = results_df.sort_values(["Variant","Classifier"]).reset_index(drop=True)

print("\nComparative Summary:", flush=True)
display(results_df)

results_df_filename_jsonl = os.path.join(model_dir, "ablation_results.jsonl")
results_df.to_json(results_df_filename_jsonl, orient='records', lines=True)
print(f"Results DataFrame saved to {results_df_filename_jsonl}", flush=True)

results_df_filename_csv = os.path.join(model_dir, "ablation_results.csv")
results_df.to_csv(results_df_filename_csv, index=False)
print(f"Results DataFrame saved to {results_df_filename_csv}", flush=True)

