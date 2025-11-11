!pip install emoji

import pandas as pd
import numpy as np
import re

from tqdm import tqdm
tqdm.pandas()

from collections import Counter
import spacy
import emoji

from time import time

from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

import string
from nltk import sent_tokenize, word_tokenize, Text
from nltk.probability import FreqDist
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ngrams

from nltk.corpus import stopwords
nltk.download('stopwords')

# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import torch

# import requests

path_ft = "/content/drive/MyDrive/Research2025/Raid/dfs_Preproc_Caract/"


df_train = pd.read_json(path_ft + 'df_train_en_clean.jsonl', orient='records', lines=True)
df_train.shape

!python -m spacy download en_core_web_sm

# Download English stopwords from NLTK
stopwords_en = set(stopwords.words('english'))

stopwords_en

nlp = spacy.load("en_core_web_sm")

path_en = "/content/drive/MyDrive/PANClef2025/Task1/Data/"
df_conect = pd.read_csv(path_en + 'Ingles.csv')
conjunctions_list = df_conect['CONECTOR'].tolist()

!pip install textstat

import textstat

def flesch_score(text, lang="en"):
    """Calculates the Flesch score."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return None  # Avoids errors if the text is empty

    num_sentences = len(sentences)
    num_words = sum(len(s.split()) for s in sentences)
    num_syllables = sum(textstat.syllable_count(word) for word in text.split())

    asl = num_words / num_sentences  # Average sentence length
    asw = num_syllables / num_words  # Average syllables per word

    #score = 206.84 - (1.02 * asl) - (60 * asw)  #FOR SPANISH TEXT
    score = 206.835 - (1.015 * asl) - (84.6 * asw)  #FOR ENGLISH TEXT

    return round(score, 2)

import math

def lexical_entropy(text):
    """Calculates lexical entropy (word variety)."""
    words = text.lower().split()
    word_counts = Counter(words)
    total_words = sum(word_counts.values())
    entropy = -sum((count / total_words) * math.log2(count / total_words) for count in word_counts.values())
    return round(entropy, 2)

def syntactic_pattern_repetition(text):
    """Calculates the repetition of syntactic patterns."""
    doc = nlp(text)

    dep_patterns = [token.dep_ for token in doc]
    dep_counts = Counter(dep_patterns)

    # Repetition index: how many times the most frequent syntactic dependencies are repeated
    repetition_index = max(dep_counts.values()) / len(dep_patterns) if dep_patterns else 0
    return round(repetition_index, 2)

# Download list of common English words
nltk.download("words")
from nltk.corpus import words

# List of common English words (for unusual word frequency)
common_words_en = set(words.words())
common_words_en

def unusual_word_frequency(text):
    """Calcula la frecuencia de palabras inusuales comparadas con un corpus de palabras comunes."""
    words_in_text = set(word.lower() for word in text.split())
    common_set = common_words_en
    unusual_words = words_in_text - common_set
    return round(len(unusual_words) / len(words_in_text), 2) if words_in_text else 0

def extraer_caracteristicas(texto):
    doc = nlp(texto)

    palabras = [token.text.lower() for token in doc if token.is_alpha] # Creates a list of lowercase words, excluding numbers and symbols. token.is_alpha filters only words made of letters.
    oraciones = list(doc.sents) # returns a list of sentences in the processed text

    # Lexical Features
    num_palabras = len(palabras)
    num_unicas = len(set(palabras))
    riqueza_lexica = num_unicas / num_palabras if num_palabras > 0 else 0 # lexical diversity, proportion of unique words over the total number of words.
    longitud_media_palabra = np.mean([len(word) for word in palabras]) if palabras else 0  # Calculates the average word length
    stopwords = [token.text.lower() for token in doc if token.is_stop] # Extracts lowercase stopwords.
    proporción_stopwords = len(stopwords) / num_palabras if num_palabras > 0 else 0 # Calculates the proportion of stopwords in the text

    # Syntactic Features
    num_oraciones = len(oraciones)
    # Evaluates whether the text uses short or long sentences
    longitud_media_oracion = np.mean([len(ora.text.split()) for ora in oraciones]) if num_oraciones > 0 else 0  # For each sentence, counts the number of words (ora.text.split()). Calculates the average words per sentence.
    # Indicates whether the text has a simple or elaborate syntactic structure
    complejidad_sintactica = sum(1 for token in doc if token.dep_ in ["acl", "advcl", "ccomp"]) / num_oraciones if num_oraciones > 0 else 0
    # Iterates through each token in the document (for token in doc).
    # Counts how many tokens have complex syntactic dependencies:
    # "acl" → Adjectival clause.
    # "advcl" → Adverbial clause.
    # "ccomp" → Complement clause.
    # Divides the total by the number of sentences to get a syntactic complexity index.


    # POS tags
    # Detection of generated text: AI models may produce POS distributions that differ from human texts.
    # analyzes the part-of-speech tags (POS tags) in the text and calculates their proportions.

    # Iterates through each token in the document (doc).
    # Extracts its grammatical tag (token.pos_), such as NOUN, VERB, ADJ, etc.
    # Uses Counter to count how many times each category appears.
    pos_tags = Counter([token.pos_ for token in doc])
    # For each POS tag, calculates its proportion in the text:
    # proportion = number of times the tag appears / total number of words
    # If num_palabras == 0, avoids division by zero.
    # Stores the proportions in a dictionary with keys like "prop_NOUN", "prop_VERB", etc.
    proporciones_pos = {f"prop_{tag}": pos_tags[tag] / num_palabras for tag in pos_tags if num_palabras > 0}
    # example
    # "The black cat runs fast."
    # "prop_DET": 0.2,  # 1 out of 5 words is a determinant (The)
    # "prop_NOUN": 0.4, # 2 out of 5 words are nouns (cat, black)
    # "prop_VERB": 0.2, # 1 out of 5 is a verb (runs)
    # "prop_ADV": 0.2,  # 1 out of 5 is an adverb (fast)


    # Punctuation marks
    # Detection of generated text: AI models may generate punctuation differently from human texts.
    signos_puntuacion = Counter([token.text for token in doc if token.is_punct])  # Counts punctuation marks in the text:
    proporciones_puntuacion = {f"punct_{p}": signos_puntuacion[p] / num_palabras for p in signos_puntuacion if num_palabras > 0} # Calculates the proportion of each punctuation mark:
    # For each mark in signos_puntuacion, calculates its proportion:
    # proportion = number of times the mark appears / total number of words
    # If num_palabras == 0, avoids division by zero.
    # Stores the proportions in a dictionary with keys like "punct_.", "punct_,", "punct_?", etc.
    # example: "Hello, how are you? Fine, thanks."
    # "punct_,": 0.2,   # 2 commas in a text of 10 words → 20%
    # "punct_¿": 0.1,   # 1 opening question mark → 10%
    # "punct_?": 0.1,   # 1 closing question mark → 10%
    # "punct_.": 0.1    # 1 full stop → 10%

    # Semantic Features
    # Detection of generated text: Some AI models may produce texts with abnormal polarity and subjectivity.
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity  # (-1 to 1, negative to positive) # Semantic, Calculates sentiment polarity
    # Returns a value between -1 and 1:
    # -1 → Negative sentiment (e.g., "Horrible, detestable, terrible").
    # 0 → Neutral sentiment (e.g., "It is an object.").
    # 1 → Positive sentiment (e.g., "Wonderful, excellent, amazing")
    subjetividad = blob.sentiment.subjectivity  # (0 to 1, objective to subjective) # semantic
    # Returns a value between 0 and 1:
    # 0 → Objective text (fact-based, no opinions).
    # 1 → Subjective text (contains opinions or emotions).

    # Diversity of bigrams and trigrams ---> syntactic analysis
    # Detection of generated text: AI models may generate texts with less n-gram diversity than human-written texts.
    # Handle empty vocabulary
    try:
        vectorizer = CountVectorizer(ngram_range=(2, 3))
        ngramas = vectorizer.fit_transform([texto])
        num_ngramas = len(vectorizer.get_feature_names_out())
        diversidad_ngramas = num_ngramas / num_palabras if num_palabras > 0 else 0
    except ValueError:
        # If vocabulary is empty, set diversity to 0
        diversidad_ngramas = 0

    sentence_lengths = [len(sentence) for sentence in doc.sents]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0

    num_tokens = len([word for word in doc if word.text not in string.punctuation])
    num_sentences = len(list(doc.sents))
    avg_word_length = sum(len(word.text) for word in doc if word.text not in string.punctuation) / num_tokens if num_tokens > 0 else 0

    #-----------------------
    # Calculates the density of unigrams, bigrams, and trigrams in a text.

    n_max=3
    #Tokenize the text into words
    tokens = nltk.word_tokenize(texto.lower())
    num_tokens = len(tokens)
    densities = {}

    for n in range(2, n_max + 1):
        n_grams = list(ngrams(tokens, n))
        densities[f"{n}-grams"] = len(n_grams) / num_tokens if num_tokens > 0 else 0


    # -------------------
    # Remove punctuation.
    text2 = ''.join(c for c in texto.lower() if not c in '"#$%&\'()*+-/:<=>@[\\]^_`{|}~')
    connector_count = {}
    total_connectors = 0

    for phrase in conjunctions_list:
      # Count occurrences of each phrase in the text.
      connector_count[phrase] = text2.count(phrase.lower())  # Ensure case-insensitive comparison
      total_connectors += connector_count[phrase]

    # Proportion of connectors with respect to the total number of words to obtain a relative measure of the "density" of connectors in the text.
    word_count = len([token.text for token in doc if not token.is_punct])  # Total number of words
    # connector_density = total_connectors / word_count
    # Check if word_count is 0 before dividing to avoid ZeroDivisionError
    connector_density = total_connectors / word_count if word_count else 0

    # Variety of connectors: Measure how many different connectors are used in the text.
    # AI-generated texts tend to be more repetitive in the use of connectors, while humans tend to vary more.
    unique_connectors = len([key for key, value in connector_count.items() if value > 0])

    return {
        "riqueza_lexica": riqueza_lexica, # --> semantic
        "longitud_media_palabra": longitud_media_palabra,
        "proporción_stopwords": proporción_stopwords,
        "longitud_media_oracion": longitud_media_oracion,
        "complejidad_sintactica": complejidad_sintactica,
        "polaridad": polaridad,
        "subjetividad": subjetividad,
        "diversidad_ngramas": diversidad_ngramas,
        "avg_sentence_length": avg_sentence_length, # f2
        "sentence_lengths": sentence_lengths, # f2
        "num_tokens": num_tokens, # f2
        "num_sentences": num_sentences, # f2
        "avg_word_length": avg_word_length, # f2
        "total_connectors": total_connectors, # f3
        "connector_density" : connector_density, # f3
        "unique_connectors" : unique_connectors, # f3
        "Flesch Score": flesch_score(texto), # f4
        "Lexical Entropy": lexical_entropy(texto), # f4
        "Syntactic Repetition": syntactic_pattern_repetition(texto), # f4
        "Unusual Word Frequency": unusual_word_frequency(texto), # f4
        **connector_count, # f3
        **proporciones_pos, # f3
        **densities, # f3
        **proporciones_puntuacion
    }

# Example of use
texto = "This article discusses the importance of machine learning. It is a fascinating topic."
caracteristicas = extraer_caracteristicas(texto)
print(caracteristicas)



# Apply the function to the 'text' column with a progress bar
tic = time()
df_train_features1 = df_train['cleaned_text'].progress_apply(extraer_caracteristicas).apply(pd.Series)
toc = time()
print()
print(f"Tiempo de ejecución: {toc - tic:.2f} segundos")

df_train_features1.shape

!pip freeze > requirements.txt



