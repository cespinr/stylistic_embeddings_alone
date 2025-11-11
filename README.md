# stylistic_embeddings_alone
# Stylistic Embeddings Alone: A Simple Approach to AI Text Detection

A minimalist approach to AI-generated text detection that challenges the assumption that detection requires architectural complexity.

## Key Findings

- **Style embeddings alone** achieve F1 scores of 0.8875 (SemEval-2024) and 0.9287 (RAID)
- A single feature type (style embeddings) captures **96.6-98.5%** of the discriminative power of complex 4-feature systems.
- Outperform fine-tuned RoBERTa-base by up to **3.7 percentage points**
- Achieve within **3.6%** of state-of-the-art competition winners
- Deliver **125-500× parameter reduction** vs. ensemble LLMs

## Approach

We systematically evaluate four feature types across two large-scale benchmarks:
- **StyleDistance embeddings** (768-dim): Content-independent stylistic signatures
- **TF-IDF n-grams** (3000-dim): Local lexical patterns
- **Linguistic features** (36-37-dim): Phraseological, syntactic, semantic metrics
- **Perplexity** (1-dim): GPT-2-based text predictability

Statistical analysis via Wilcoxon tests reveals that **style embeddings capture most discriminative signals**, while perplexity contributes negligibly.

## Repository Structure
```
├── Code/
│   ├── ablation.py                    # Ablation study: 27 configs (3 classifiers × 9 feature combinations)
│   ├── feature_selection.py           # RidgeCV selection: 197/212 → 36/37 linguistic features
│   ├── linguistic_feat.py             # Extract phraseological, syntactic, semantic features (spaCy)
│   ├── perplex_tfidf_style_feat.py    # Extract Perplexity (GPT-2) + TF-IDF (cuML) + StyleDistance (768-dim)
│   ├── requirements_consolidated.txt  # Python dependencies
│   └── statistical_significance.py    # Wilcoxon tests with Bonferroni correction (α=0.0033)
└── README.md
```

### Quick Descriptions

| File | Purpose | Input → Output |
|------|---------|----------------|
| `perplex_tfidf_style_feat.py` | Extract deep learning features | Text → Perplexity (1-dim) + TF-IDF (3000-dim) + Style (768-dim) |
| `linguistic_feat.py` | Extract statistical features | Text → 197/212 linguistic features |
| `feature_selection.py` | Reduce dimensionality | 197/212 features → 36/37 selected features (RidgeCV) |
| `ablation.py` | Run experiments | Features → 27 configurations × performance metrics |
| `statistical_significance.py` | Validate results | Top-6 configs → p-values + significance tests |


## Datasets

### SemEval-2024 Task 8 (Subtask A)
- **Size**: 119,757 train / 29,939 val / 35,184 test
- **Domains**: Wikipedia, Reddit, arXiv, PeerRead, Creative Writing
- **Generators**: GPT-3.5, GPT-4, Claude, Cohere, Llama-2

### RAID
- **Size**: ~250K balanced subset (from 6M+ total)
- **Generators**: 11 LLMs (GPT-2/3/4, ChatGPT, Llama-2, Cohere, MPT-30B, Mistral 7B, etc.)
- **Genres**: ArXiv, Recipes, Reddit, Books, NYT, Poetry, IMDb, Wikipedia
- **Decoding**: 4 strategies (Greedy, Sampling, ±repetition penalty)

## 📝 Citation
```bibtex
```

## Authors

- **César Espin-Riofrio** - University of Guayaquil - cesar.espinr@ug.edu.ec
- **Jenny Ortiz-Zambrano** - University of Guayaquil - jenny.ortizz@ug.edu.ec
- **Arturo Montejo-Ráez** - University of Jaen - amontejo@ujaen.es

## Acknowledgments

- University of Guayaquil
- University of Jaen
- SemEval-2024 Task 8 organizers
- RAID benchmark creators
