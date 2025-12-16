# stylistic_embeddings_alone
# Stylistic Embeddings Alone: A Simple Approach to AI Text Detection

A minimalist approach to AI-generated text detection that challenges the assumption that detection requires architectural complexity.

## Key Findings

- **Style embeddings alone** achieve F1 scores of 0.8875 (SemEval-2024) and 0.9287 (RAID)
- A single feature type (style embeddings) captures **96.6-98.5%** of the discriminative power of complex 4-feature systems
- Outperform fine-tuned RoBERTa-base by **2.4-5.0 percentage points**
- Achieve within **3.4%** of state-of-the-art competition winners
- Substantially exceed **Binoculars zero-shot detection** by **4.8-15.1 points** (Style Only) and **7.9-16.6 points** (All Features)
- Deliver **100-500× parameter reduction** vs. ensemble LLMs

## Approach

We systematically evaluate **four distinct approaches** across two large-scale benchmarks:

### Feature-Based Methods (with ablation studies):
1. **StyleDistance embeddings** (768-dim): Content-independent stylistic signatures trained on 40 distinct aspects (formality, verbosity, syntactic complexity, discourse structure)
2. **TF-IDF n-grams** (3000-dim): Local lexical patterns
3. **Linguistic features** (36-37-dim): Phraseological, syntactic, semantic metrics
4. **Perplexity** (1-dim): GPT-2-based text predictability

### Baseline Comparisons:
5. **RoBERTa-base** (125M params): Fine-tuned transformer baseline
6. **Binoculars**: Zero-shot detection using observer-performer perplexity differences (falcon-7b models)

Statistical analysis via Wilcoxon tests reveals that **style embeddings capture most discriminative signals**, while perplexity contributes negligibly when combined with other features.

## Results Summary

### SemEval-2024 Task 8
| Method | F1 Score | vs RoBERTa | vs Binoculars |
|--------|----------|------------|---------------|
| All Features (RF) | 0.9190 | +2.4 pts | +7.9 pts |
| Style Only (RF) | 0.8875 | -0.7 pts | +4.8 pts |
| RoBERTa-base | 0.8946 | — | +5.5 pts |
| Binoculars | 0.8400 | -5.5 pts | — |

### RAID
| Method | F1 Score | vs RoBERTa | vs Binoculars |
|--------|----------|------------|---------------|
| All Features (RF) | 0.9425 | +5.0 pts | +16.6 pts |
| Style Only (RF) | 0.9287 | +3.7 pts | +15.1 pts |
| RoBERTa-base | 0.8922 | — | +13.2 pts |
| Binoculars | 0.7600 | -13.2 pts | — |

**Key Finding:** StyleDistance alone substantially exceeds zero-shot state-of-the-art (Binoculars) while offering interpretability through explicit feature inspection.

## Trade-offs

### Advantages:
- **100-500× smaller model footprint** (<1M vs 125-500M parameters for classifiers)
- **Explicit interpretability** through feature-level inspection
- **Modular architecture** enabling selective feature extraction
- **Superior accuracy** vs both RoBERTa and zero-shot methods

### Limitations:
- **14.7-72.6× slower inference** due to feature extraction overhead
- Best suited for **batch processing** and **interpretability-critical scenarios** (academic integrity, legal compliance)
- **Not suitable** for real-time applications requiring low latency

## Repository Structure
```
├── Code/
│   ├── ablation.py                    # Ablation study: 27 configs (3 classifiers × 9 feature combinations)
│   ├── feature_selection.py           # RidgeCV selection: 197/212 → 36/37 linguistic features
│   ├── linguistic_feat.py             # Extract phraseological, syntactic, semantic features (spaCy)
│   ├── perplex_tfidf_style_feat.py    # Extract Perplexity (GPT-2) + TF-IDF (cuML) + StyleDistance (768-dim)
│   ├── binoculars.py                  # Zero-shot detection using Binoculars (falcon-7b observer/performer)
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
| `binoculars.py` | Zero-shot baseline evaluation | Text → Binoculars score + prediction (0/1) |
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

## Deployment Scenarios

| Scenario | Feature-based | RoBERTa | Binoculars | Key Consideration |
|----------|---------------|---------|------------|-------------------|
| Batch processing | ✓✓ | ✓ | ✓✓ | High throughput acceptable when accuracy matters |
| Interpretability-critical | ✓✓ | × | × | Explicit features justify slower inference |
| Resource-constrained | ✓✓ | × | ✓✓ | Smaller models enable edge deployment |
| Real-time applications | × | ✓✓ | ✓✓ | Low latency required |
| Zero-shot deployment | × | × | ✓✓ | No training data available |

**✓✓ = Highly suitable; ✓ = Suitable; × = Not suitable**

## 📄 Citation


## Authors


## Acknowledgments


## Funding


