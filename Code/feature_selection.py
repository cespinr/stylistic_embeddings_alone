import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV

path_ft = "/content/drive/MyDrive/Research2025/Raid/dfs_Preproc_Caract/"

df_train_clean = pd.read_json(path_ft + 'df_train_en_clean.jsonl', orient='records', lines=True)
df_test_clean = pd.read_json(path_ft + "df_val_en_clean.jsonl", orient='records', lines=True)

df_train_clean

df_train_ft1 = pd.read_json(path_ft + "df_train_en_features1.jsonl", orient='records', lines=True)
df_train_ft2 = pd.read_json(path_ft + "df_train_en_features2.jsonl", orient='records', lines=True)
df_train_ft3 = pd.read_json(path_ft + "df_train_en_features3.jsonl", orient='records', lines=True)
df_train_ft4 = pd.read_json(path_ft + "df_train_en_features4.jsonl", orient='records', lines=True)

df_test_ft1 = pd.read_json(path_ft + "df_val_en_features1.jsonl", orient='records', lines=True)
df_test_ft2 = pd.read_json(path_ft + "df_val_en_features2.jsonl", orient='records', lines=True)
df_test_ft3 = pd.read_json(path_ft + "df_val_en_features3.jsonl", orient='records', lines=True)
df_test_ft4 = pd.read_json(path_ft + "df_val_en_features4.jsonl", orient='records', lines=True)


print(df_train_clean.shape)
print(df_test_clean.shape)

print(df_train_ft1.shape)
print(df_test_ft1.shape)

print(df_train_ft2.shape)
print(df_test_ft2.shape)

print(df_train_ft3.shape)
print(df_test_ft3.shape)

print(df_train_ft4.shape)
print(df_test_ft4.shape)

df_train = pd.concat([df_train_clean, df_train_ft1, df_train_ft2, df_train_ft3, df_train_ft4],  axis=1)
df_test = pd.concat([df_test_clean, df_test_ft1, df_test_ft2, df_test_ft3, df_test_ft4],  axis=1)

df_train

X = df_train.iloc[:, 13:]
y = df_train['label']

X

# Step 1: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# Step 2: Train Random Forest and select features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Selection using the mean of importances as threshold
selector = SelectFromModel(rf, threshold="mean", prefit=True)

# Apply selection
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

print(f"Features selected: {X_train_sel.shape[1]} / {X.shape[1]}")

# Step 3: Refine with RidgeCV
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sel)
X_test_scaled = scaler.transform(X_test_sel)

# Impute NaN values using SimpleImputer before fitting RidgeCV
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train_scaled)
X_test_imputed = imputer.transform(X_test_scaled)

ridge = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
ridge.fit(X_train_imputed, y_train)

# View coefficients:
coefs = np.abs(ridge.coef_)
coef_threshold = np.percentile(coefs, 25)  # remove the lowest 25%
mask = coefs > coef_threshold

X_train_final = X_train_imputed[:, mask]
X_test_final = X_test_imputed[:, mask]

print(f"Features after RidgeCV: {X_train_final.shape[1]}")

# Step 4: Final classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_final, y_train)

# Evaluate
acc = clf.score(X_test_final, y_test)
print(f"Final Accuracy: {acc:.4f}")

# Get feature importances from the trained Random Forest model
importances = rf.feature_importances_

# Get the names of the original features
feature_names = X.columns

# Create a DataFrame of feature importances
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

# Sort the features by importance
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

# Select only the features that were selected by the SelectFromModel
selected_feature_mask = selector.get_support()
selected_features_df = feature_importances_df[feature_importances_df['feature'].isin(feature_names[selected_feature_mask])].copy()

# Create a dictionary for Spanish to English translation of feature names
translation_dict = {
    'riqueza_lexica': 'Lexical Richness',
    'longitud_media_palabra': 'Avg. Word Length',
    'proporción_stopwords': 'Stopwords Ratio',
    'longitud_media_oracion': 'Avg. Sentence Length',
    'complejidad_sintactica': 'Syntactic Complexity',
    'polaridad': 'Polarity',
    'subjetividad': 'Subjectivity',
    'diversidad_ngramas': 'n-gram Diversity',
    'prop_PUNCT': 'Punctuation Prop.',
    'prop_VERB': 'Verb Prop.',
    'prop_NOUN': 'Noun Prop.',
    'prop_ADV': 'Adverb Prop.',
    'prop_INTJ': 'Interjection Prop.',
    'prop_ADP': 'Adposition Prop.',
    'prop_SCONJ': 'Subordinating Conjunction Prop.',
    'punct_,': 'Comma Prop.',
    'punct_.': 'Period Prop.',
    'prop_SPACE': 'Space Prop.',
    'prop_X': 'Other Prop.',
    'punct_;': 'Semicolon Prop.',
    'punct_!': 'Exclamation Prop.',
    'prop_PRON': 'Pronoun Prop.',
    'punct_?': 'Question Mark Prop.',
    'num_tokens': 'Number of Tokens',
    'num_sentences': 'Number of Sentences',
    'avg_word_length': 'Avg Word Length', # Already present, but keeping for consistency
    'total_connectors': 'Total Connectors',
    'connector_density': 'Connector Density',
    'unique_connectors': 'Unique Connectors',
    'also': 'also',
    'or': 'ir',
    'so': 'so',
    '2-grams': '2-grams',
    '3-grams': '3-grams',
    'avg_depth': 'Avg. Depth',
    'Flesch Score': 'Flesch Score',
    'Lexical Entropy': 'Lexical Entropy',
    'Unusual Word Frequency': 'Unusual Word Frequency',
    'prop_PROPN': 'Proper Noun Prop.',
    'prop_NUM': 'Number Prop.',
    'prop_ADJ': 'Adjective Prop.',
    'prop_AUX': 'Auxiliary Prop.',
    'prop_CCONJ': 'Coordinating Conjunction Prop.',
    'as': 'as',
    'and': 'and',
    'If': 'if'
}

# Apply translation to the 'feature' column
selected_features_df['feature'] = selected_features_df['feature'].replace(translation_dict)


# Plot the feature importances of the selected features
plt.figure(figsize=(12, 12)) # Increased figure height for better y-axis label visibility
ax = sns.barplot(x='importance', y='feature', data=selected_features_df)
plt.title('Importance of Selected Features - RAID')
plt.xlabel('Importance')
plt.ylabel('Feature')

# Add vertical lines at x-axis tick values
for x_val in ax.get_xticks():
    plt.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.5)

plt.show()


# Get the names of the selected features
#selected_feature_names = X_encoded.columns[selector.get_support()]
#selected_feature_names = selected_feature_names[mask]
selected_feature_indices = np.where(mask)[0]
selected_feature_names = X.columns[selector.get_support()][selected_feature_indices]

# Create a new DataFrame with the selected features
df_train_selected = pd.DataFrame(X_train_final, columns=selected_feature_names)
df_test_selected = pd.DataFrame(X_test_final, columns=selected_feature_names)

print(df_train_selected.head())
print(df_test_selected.head())

selected_feature_names

X_new = X[selected_feature_names]

X_train_new = pd.concat([df_train.iloc[:, :13], X_new], axis=1)

X_test_new = pd.concat([df_test.iloc[:, :13], df_test[selected_feature_names]], axis=1)


# Check if X_train_new_features and X_val_new_features have the same columns
print(X_train_new.columns.tolist() == X_test_new.columns.tolist())

!pip freeze > requirements.txt