#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Comparative Evaluation of Text Semantic Similarity Methods Using Classical and Transformer-Based Representations],
  abstract: [
    This study benchmarks four sentence-similarity methods: TF-IDF cosine similarity, averaged Word2Vec embeddings, pretrained SBERT, and a fine-tuned SBERT variant across five complementary benchmarks.
  ],
  authors: (
    (
      name: "Maksym Prots",
      // department: [Co-Founder],
      organization: [61975],
      location: [Poland, Warsaw],
      email: "imaxprots@gmail.com",
    ),
    (
      name: "Example",
      // department: [Co-Founder],
      // organization: [Typst GmbH],
      location: [Poland, Warsaw],
      email: "example@example.com",
    ),
  ),
  index-terms: (
    "Semantic Textual Similarity",
    "Paraphrase Detection",
    "Natural Language Processing",
    "SyntaxTransformer Models",
    "Sentence-BERT",
    "Semantic Analysis",
    "Machine Learning",
    "Deep Learning",
    "Text Classification",
  ),
  bibliography: bibliography("refs.bib"),
  // figure-supplement: [Fig.],
)

= Introduction

Semantic textual similarity (STS) is at the core of various natural language processing applications, including machine translation quality estimation, duplicate question detection, information retrieval, and conversational AI. Accurately quantifying how closely two text spans align in meaning requires representations that capture lexical overlap, syntactic structure, and contextual semantics. Over the past decade, the field has changed from classical bag-of-words and term-frequency models to distributed word embeddings, and more recently to contextualized transformer architectures. Despite this rapid evolution, systematic comparisons that place classical, shallow neural, and deep transformer-based methods on a unified evaluation framework remain limited.

This paper presents a comprehensive comparative evaluation of text semantic similarity methods spanning three representational paradigms: (1) classical statistical representations (TF-IDF), (2) shallow distributed representations (Word2Vec), and (3) transformer-based contextual embeddings (Sentence-BERT and fine-tuned variants). We additionally incorporate BERTScore as a token-level neural reference metric. Our evaluation covers both regression benchmarks (STS-B, SICK-R, STS-Annual) and binary classification tasks (MRPC, QQP), employing Pearson correlation, Spearman rank correlation, accuracy, and F1-score as primary evaluation metrics. Statistical significance is assessed via Fisher z-transformation. Our contributions are: (i) a unified experimental pipeline comparing classical and transformer-based similarity methods under identical evaluation conditions; (ii) an analysis of how fine-tuning on task-specific data impacts representation quality; and (iii) a qualitative error analysis highlighting failure modes across representational families.

= Related Work

Early approaches to semantic similarity relied on lexical overlap (e.g., Jaccard, overlap coefficient) and term weighting schemes such as TF-IDF or BM25. Cosine similarity in vector space models provided strong baselines but inherently ignore word order, syntax, and contextual polysemy.

The introduction of static distributed representations via Word2Vec, GloVe, and FastText marked a shift toward learning dense vector spaces where semantic relationships are captured through local co-occurrence statistics. While these models improved performance on synonymy and paraphrase detection, they remain context-agnostic, each word has a single fixed vector, struggling with polysemy, compositional semantics, and negation.

The advent of transformer architectures, particularly BERT (2018), revolutionized representation learning by introducing bidirectional contextual embeddings, enabling dynamic word representations conditioned on surrounding tokens. However, even early transformers (e.g., BERT-base) are not strictly “bidirectional” in a mathematical sense across all layers, but they do condition on left and right context simultaneously via masked language modeling.

Sentence-BERT (SBERT) adapted this paradigm to sentence-level similarity by employing a siamese or twin network architecture optimized with cosine similarity or triplet loss, drastically reducing inference latency compared to cross-encoder baselines (e.g., BERT fine-tuned on NLI or STS with cross-attention).

BERTScore further refined token-level alignment by leveraging contextual embeddings and soft alignment via BLEU/ROUGE-style matching. Benchmark datasets such as STS-B, SICK-R, and GLUE (MRPC, QQP) have become standard for evaluating similarity and paraphrase detection.

= Methodology


== Overview

Our experimental framework evaluates five similarity estimation strategies across regression and classification benchmarks. All models are evaluated under identical preprocessing, scoring, and thresholding conditions to ensure fair comparison. Reproducibility is enforced via a fixed random seed (RANDOM_SEED=42) across NumPy and PyTorch backends.

== TF-IDF Representation

Classical TF-IDF representations model text as sparse term-frequency vectors weighted by inverse document frequency. Sentence pairs are vectorized independently, and similarity is computed as the cosine similarity between their respective TF-IDF vectors. This baseline captures lexical overlap and term rarity but disregards syntactic ordering and contextual meaning.

== Word2Vec Representation

We employ a pre-trained Word2Vec model to map tokens to static dense embeddings. Sentence representations are constructed by averaging the token embeddings, optionally weighted by IDF scores to mitigate the influence of frequent function words. Similarity is again measured via cosine similarity in the averaged embedding space. While this approach captures distributional semantics, it remains vulnerable to context-independent word sense ambiguity.

== Sentence-BERT (SBERT)

We evaluate two variants of Sentence-BERT: (1) a pretrained model fine-tuned on natural language inference (NLI) data, and (2) a task-specific variant fine-tuned directly on STS-B using cosine similarity loss. Both utilize a siamese transformer architecture that encodes sentences independently before computing cosine similarity. The fine-tuned variant is optimized via a DataLoader with batched training, allowing the model to adapt its representation space to the continuous similarity distribution of the target benchmark.

== Similarity Computation

For vector-based methods (TF-IDF, Word2Vec, SBERT), similarity is computed as the cosine similarity between encoded sentence representations. BERTScore is evaluated as a token-level neural reference metric, leveraging dynamic alignment over contextualized token embeddings without requiring sentence-level pooling. All continuous similarity scores are normalized to [0,1] where applicable (e.g., STS-B raw scores divided by 5.0) to ensure metric comparability across datasets.

= Datasets

- STS-B @huggingface:dataset:stsb_multi_mt — continuous human similarity ratings
- SICK-R @marelli-etal-2014-sick — relatedness scores + entailment labels
- STS-Annual @huggingface:dataset:stsb_multi_mt — multi-domain SemEval shared-task pairs
- MRPC @wang2019glue — binary paraphrase identification (news)
- QQP @wang2019glue — large-scale duplicate question detection

STS-B, SICK-R, and STS-Annual are evaluated as regression tasks using Pearson correlation (r) and Spearman rank correlation (p). MRPC and QQP are treated as binary classification tasks using Accuracy and F1-score. Continuous scores are normalized to [0,1] prior to thresholding. STS-B and SICK are regression benchmarks (Pearson r, Spearman p). STS-Annual extends the regression suite across six heterogeneous domains (headlines, forum answers, image captions, Twitter, student answers). MRPC and QQP are binary classification benchmarks (Accuracy, F1). A fifth method, BERTScore, is included as a token-level neural reference metric. Statistical significance between model pairs is assessed via Fisher z-transformation.

= Experimental Setup

All experiments are conducted in a unified Python pipeline. Continuous similarity scores are evaluated directly against human annotations using Pearson and Spearman correlations. For classification benchmarks (MRPC, QQP), an optimal decision threshold is determined via grid search on the training split, and evaluation metrics are computed on the held-out test set. Model rankings are generated by sorting on F1-score, with results tabulated for reproducibility.

Statistical significance between model pairs on regression benchmarks is assessed using the Fisher z-transformation on Pearson correlation coefficients, with significance reported at p < 0.05. Visualization of cross-dataset performance is generated via multi-panel bar charts comparing regression and classification metrics across all five benchmarks.

= Results

== STS-B & Semantic Similarity Benchmarks (STS-B, SICK, STS-Annual)

On the STS-B regression benchmark, transformer-based methods consistently outperform classical and static embedding baselines.
TF-IDF achieves a Pearson $r$ of 0.690, reflecting its reliance on surface lexical overlap.
Word2Vec performs poorly on STS-B ($r=0.109$), indicating that static embeddings fail to capture fine-grained semantic similarity when lexical variation is high.
The pretrained SBERT model reaches $r=0.838$, while the STS-B fine-tuned SBERT variant achieves r=0.868, demonstrating the benefit of task-specific optimisation.
BERTScore underperforms on STS-B ($r=0.569$) compared to SBERT, despite its token-level alignment mechanism.

Similar trends hold for SICK and STS-Annual.
On SICK, fine-tuned SBERT achieves r=$0.860$ (pretrained: $0.857$), Word2Vec reaches only 0.385, and BERTScore reaches 0.655.

On STS-Annual, fine-tuned SBERT again leads with r=0.868, while Word2Vec collapses to 0.109, the domain shift penalises static embeddings severely.

== Classification Benchmarks (MRPC & QQP)

Threshold optimisation on the training splits yields strong classification performance for contextual models.

On MRPC, fine-tuned SBERT achieves an F1-score of 0.837 (optimal threshold ~0.72), outperforming pretrained SBERT (0.832) and Word2Vec (0.812).
Notably, TF-IDF performs competitively on MRPC (F1 = 0.821) due to high lexical overlap in the dataset.
BERTScore reaches 0.822 F1, slightly below fine-tuned SBERT.

On QQP, the gap between classical and transformer methods widens due to the dataset's scale and lexical variation.
Fine-tuned SBERT reaches 0.739 F1, while pretrained SBERT achieves 0.737.
TF-IDF drops to 0.641 and Word2Vec to 0.559, confirming that lexical and static methods cannot handle paraphrastic variance in question matching.
Accuracy trends mirror F1.

== Cross-Dataset Performance

Aggregated results across all five benchmarks confirm a consistent hierarchy:

Fine-tuned SBERT > Pretrained SBERT ≈ BERTScore > TF-IDF > Word2Vec

The performance gap is most pronounced on STS-Annual and QQP, where domain shift and lexical divergence penalise static and lexical baselines.
Statistical testing (Fisher z-transformation) confirms that improvements from Word2Vec to SBERT are significant ($p < 0.01$) across all regression datasets, despite Word2Vec's near-random performance on STS-Annual.

== Discussion of Results

= Error Analysis

= Discussion

= Conclusion and Future Work