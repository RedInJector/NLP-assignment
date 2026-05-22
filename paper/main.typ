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
      // organization: [Typst GmbH],
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

= Related Work

= Methodology

== Overview

== TF-IDF Representation

== Word2Vec Representation

== Sentence-BERT (SBERT)

== Similarity Computation

= Datasets

- STS-B @huggingface:dataset:stsb_multi_mt — continuous human similarity ratings
- SICK-R @marelli-etal-2014-sick — relatedness scores + entailment labels
- STS-Annual @huggingface:dataset:stsb_multi_mt — multi-domain SemEval shared-task pairs
- MRPC @wang2019glue — binary paraphrase identification (news)
- QQP @wang2019glue — large-scale duplicate question detection

STS-B and SICK are regression benchmarks (Pearson r, Spearman ρ). STS-Annual extends the regression suite across six heterogeneous domains (headlines, forum answers, image captions, Twitter, student answers). MRPC and QQP are binary classification benchmarks (Accuracy, F1). A fifth method, BERTScore, is included as a token-level neural reference metric. Statistical significance between model pairs is assessed via Fisher z-transformation.

= Experimental Setup

= Results

== STS-B Results

TF-IDF vs Word2Vec vs SBERT

tables and stuff

== Something else

== Discussion of Results

= Error Analysis

= Discussion

= Conclusion and Future Work