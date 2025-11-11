import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.nn import functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
from tqdm import tqdm
tqdm.pandas()
import random
import joblib
import os


import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

path_ft = "/content/drive/MyDrive/Research2025/Raid/dfs_Preproc_Caract/"

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU.")

# FOR PERPLEXITY

df_train = pd.read_json(path_ft + 'df_train_preprocessed_for_perplexity.jsonl', orient='records', lines=True)

# Sample 10% of the training data, ensuring both classes are included proportionally
df_train_sampled = df_train.groupby('label').apply(lambda x: x.sample(frac=0.01, random_state=42)).reset_index(drop=True)

# Shuffle the sampled data
df_train_sampled = df_train_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sampled {len(df_train_sampled)} training records with the following class distribution:")
print(df_train_sampled['label'].value_counts())

class PerplexityDetector:
    def __init__(self, model_name='gpt2'):
        """
        Initialize the perplexity detector with a pre-trained language model.

        Args:
            model_name (str): Name of the model to use for perplexity calculation
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_perplexity(self, text, max_length=512):
        """
        Calculate perplexity for a given text.

        Args:
            text (str): Input text

        Returns:
            float: Perplexity score
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            truncation=True,
            padding=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)

        return perplexity.item()

    def calculate_batch_perplexity(self, texts, batch_size=8, max_length=512):
        """
        Calculate perplexity for a batch of texts.

        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for processing
            max_length (int): Maximum sequence length

        Returns:
            list: List of perplexity scores
        """
        perplexities = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Calculating perplexities"):
            batch_texts = texts[i:i+batch_size]

            # Tokenize batch
            encodings = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                max_length=max_length,
                truncation=True,
                padding=True
            )

            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)

                # Calculate perplexity for each sample in batch
                for j in range(len(batch_texts)):
                    # Get loss for individual sample
                    sample_input_ids = input_ids[j:j+1]
                    sample_attention_mask = attention_mask[j:j+1]

                    sample_outputs = self.model(
                        sample_input_ids,
                        attention_mask=sample_attention_mask,
                        labels=sample_input_ids
                    )

                    loss = sample_outputs.loss
                    perplexity = torch.exp(loss).item()
                    perplexities.append(perplexity)

        return perplexities

# Use the preprocessed training dataframe
train_texts = df_train_sampled['preprocessed_text_for_perplexity'].tolist()
train_labels = df_train_sampled['label'].tolist()

# Use the preprocessed test dataframe
#test_texts = df_test['preprocessed_text_for_perplexity'].tolist()
#test_labels = df_test['label'].tolist()

# Initialize detector
detector = PerplexityDetector(model_name='gpt2')

# Calculate perplexity for training data
print("Calculating perplexities for training data...")
train_perplexities = detector.calculate_batch_perplexity(train_texts)
print("\nFirst 5 training perplexities:")
print(train_perplexities[:5])
print(f"\nAverage training perplexity: {np.mean(train_perplexities):.4f}")


# Calculate perplexity for test data
#print("\nCalculating perplexities for test data...")
#test_perplexities = detector.calculate_batch_perplexity(test_texts)
#print("\nFirst 5 test perplexities:")
#print(test_perplexities[:5])
#print(f"\nAverage test perplexity: {np.mean(test_perplexities):.4f}")

# FOR TF-IDF

df_train = pd.read_json(path_ft + 'df_train_processed.jsonl', orient='records', lines=True)

# Sample 10% of the training data, ensuring both classes are included proportionally
df_train_sampled = df_train.groupby('label').apply(lambda x: x.sample(frac=0.01, random_state=42)).reset_index(drop=True)

# Shuffle the sampled data
df_train_sampled = df_train_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sampled {len(df_train_sampled)} training records with the following class distribution:")
print(df_train_sampled['label'].value_counts())

# Example using cuML for GPU-accelerated TF-IDF (requires cuML installation)

from cuml.feature_extraction.text import TfidfVectorizer as cuMLTfidfVectorizer

# Initialize cuML TF-IDF Vectorizer
cuml_tfidf_vectorizer = cuMLTfidfVectorizer(ngram_range=(1, 4), max_features=3000)

# Fit and transform on the training data (assuming df_train_sampled is a cuDF DataFrame)
print("Calculating TF-IDF features with cuML...")
start_time = time.time()
X_train_cuml = cuml_tfidf_vectorizer.fit_transform(df_train_sampled['processed_text'])
end_time = time.time()
print(f"cuML TF-IDF calculation time: {end_time - start_time:.4f} seconds")

print("Shape of cuML TF-IDF features:", X_train_cuml.shape)

#print("Note: cuML requires a CUDA-enabled GPU and may require specific installation steps.")
#print("The standard scikit-learn TfidfVectorizer used previously runs on the CPU.")



df_train = pd.read_json(path_ft + 'df_train_en_clean.jsonl', orient='records', lines=True)

# Sample 10% of the training data, ensuring both classes are included proportionally
df_train_sampled = df_train.groupby('label').apply(lambda x: x.sample(frac=0.01, random_state=42)).reset_index(drop=True)

# Shuffle the sampled data
df_train_sampled = df_train_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Sampled {len(df_train_sampled)} training records with the following class distribution:")
print(df_train_sampled['label'].value_counts())

df_train_sampled

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer('StyleDistance/styledistance') # Load model
model.to(device)

df_train['embeddings'] = df_train_sampled['cleaned_text'].progress_apply(lambda x: model.encode(x))

!pip freeze > requirements.txt