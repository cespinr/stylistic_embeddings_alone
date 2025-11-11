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
