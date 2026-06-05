#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Comparative Evaluation of Text Semantic Similarity Methods Using Classical and Transformer-Based Representations],
  abstract: [
    This study benchmarks five sentence-similarity methods: TF-IDF cosine similarity, averaged Word2Vec embeddings, pretrained SBERT, fine-tuned SBERT variant, and BERTScore across five complementary benchmarks.
  ],
  authors: (
    (
      name: "Prots Maksym",
      department: [61975],
      organization: [Vistula University],
      location: [Poland, Warsaw],
      email: "imaxprots@gmail.com",
    ),
    (
      name: "Danden Ahmed",
      department: [78459],
      organization: [Vistula University],
      location: [Poland, Warsaw],
      email: "adanden1@stu.vistula.edu.pl",
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

#set text(
  size: 11pt
)

= Introduction

Semantic textual similarity (STS) is at the core of various natural language processing applications, including machine translation quality estimation, duplicate question detection, information retrieval, and conversational AI. Accurately quantifying how closely two text spans align in meaning requires representations that capture lexical overlap, syntactic structure, and contextual semantics. Over the past decade, the field has changed from classical bag-of-words and term-frequency models to distributed word embeddings, and more recently to contextualized transformer architectures. Despite this rapid evolution, systematic comparisons that place classical, shallow neural, and deep transformer-based methods on a unified evaluation framework remain limited.

Beyond theoretical considerations, the choice of semantic similarity metric holds profound practical implications for downstream NLP applications, ranging from search engine ranking and information retrieval to customer support ticket deduplication and conversational AI safety filters. In these real-world scenarios, the decision between classical statistical representations and deep neural architectures involves a critical trade-off between computational efficiency and representational fidelity. Classical methods such as TF-IDF@salton1975vector offer immediate advantages in terms of interpretability and resource constraints; they require no GPU resources, operate exclusively on CPU, and scale linearly with corpus size, making them ideal for massive-scale retrieval tasks where latency must be minimized. In contrast, transformer-based models like SBERT@devlin-etal-2019-bert deliver superior accuracy by capturing nuanced contextual semantics, albeit at a significantly higher inference cost due to the quadratic complexity of self-attention mechanisms and the necessity of GPU acceleration. While numerous studies have benchmarked individual models in isolation, this specific trade-off—quantifying exactly how much performance gain is required to justify the substantial increase in computational overhead—remains underexplored in controlled evaluations that unify classical, shallow neural, and deep transformer-based methods under a single experimental framework. Addressing this gap is essential for guiding practitioners in selecting appropriate similarity estimators based on their specific constraints regarding budget, latency tolerance, and domain requirements.

This paper presents a comprehensive comparative evaluation of text semantic similarity methods spanning three representational paradigms: 
- Classical statistical representations (TF-IDF@salton1975vector)
- Shallow distributed representations (Word2Vec@mikolov2013distributed)
- Transformer-based contextual embeddings (BERT@devlin-etal-2019-bert variants).

We additionally incorporate BERTScore@Zhang2019BERTScoreET as a token-level neural reference metric. Our evaluation covers both regression benchmarks (STS-B@huggingface:dataset:stsb_multi_mt, SICK-R@marelli-etal-2014-sick, STS-Annual@huggingface:dataset:stsb_multi_mt) and binary classification tasks (MRPC@wang2019glue, QQP@wang2019glue), employing Pearson correlation, Spearman rank correlation, accuracy, and F1-score as primary evaluation metrics. Statistical significance is assessed via Fisher z-transformation. 



Our contributions are: (i) a unified experimental pipeline comparing classical and transformer-based similarity methods under identical evaluation conditions; (ii) an analysis of how fine-tuning on task-specific data impacts representation quality; and (iii) a qualitative error analysis highlighting failure modes across representational families.

= Related Work

Early approaches to semantic similarity relied on lexical overlap (e.g., Jaccard, overlap coefficient) and term weighting schemes such as TF-IDF@salton1975vector. Cosine similarity in vector space models provided strong baselines but inherently ignore word order, syntax, and contextual polysemy.

The introduction of static distributed representations via for example Word2Vec@mikolov2013distributed marked a shift toward learning dense vector spaces where semantic relationships are captured through local co-occurrence statistics. While these models improved performance on synonymy and paraphrase detection, they remain context-agnostic, each word has a single fixed vector, struggling with polysemy, compositional semantics, and negation.

The advent of transformer architectures, particularly BERT@devlin-etal-2019-bert, revolutionized representation learning by introducing bidirectional contextual embeddings, enabling dynamic word representations conditioned on surrounding tokens. However, even early transformers (e.g., BERT-base) are not strictly “bidirectional” in a mathematical sense across all layers, but they do condition on left and right context simultaneously via masked language modeling.

Sentence-BERT@reimers-2019-sentence-bert (SBERT) adapted this paradigm to sentence-level similarity by employing a siamese or twin network architecture optimized with cosine similarity or triplet loss, drastically reducing inference latency compared to cross-encoder baselines.

BERTScore@Zhang2019BERTScoreET further refined token-level alignment by leveraging contextual embeddings and soft alignment via BLEU/ROUGE-style matching. Benchmark datasets such as STS-B@huggingface:dataset:stsb_multi_mt, SICK-R@marelli-etal-2014-sick, and GLUE@wang2019glue (MRPC, QQP) have become standard for evaluating similarity and paraphrase detection.

= Methodology


== Overview

Our experimental framework evaluates five similarity estimation strategies across regression and classification benchmarks. All models are evaluated under identical preprocessing, scoring, and thresholding conditions to ensure fair comparison. Reproducibility is enforced via a fixed random seed (RANDOM_SEED=42) across NumPy and PyTorch backends.

=== Preprocessing and Normalization

To ensure fair comparison across diverse representation families, we apply minimal preprocessing that preserves each model's native tokenization behaviour. For **TF-IDF** and **Sentence-BERT**, we use the default tokenizers provided by `sklearn.feature_extraction.text.TfidfVectorizer` (which lowercases by default) and `sentence-transformers` (which uses WordPiece tokenization without additional lowercasing, as the pretrained model expects original casing). For **Word2Vec**, we explicitly lowercase all text and split on whitespace, because static embeddings trained on lowercased corpora are highly sensitive to case mismatch; lowercasing ensures that "The" and "the" map to the same vector. This discrepancy is a known limitation: Word2Vec loses some semantic nuance (e.g., proper noun distinction) but gains coverage.

*Score normalisation* is applied to regression datasets to map human annotations into a comparable $[0,1]$ interval. STS-B provides raw scores from 0 to 5, which we divide by 5.0. SICK-R provides relatedness scores from 1 (completely unrelated) to 5 (completely related); we apply $"score_norm" = ("score" - 1) / 4.0$, which preserves the relative distance while anchoring 0 as minimal similarity. For MRPC and QQP, the labels are binary (0 = non-paraphrase / non-duplicate, 1 = paraphrase / duplicate). These require no normalisation, as they are already in ${0,1}$. Threshold tuning (Section 3.4) is applied directly to raw similarity scores produced by each model.

All five datasets contain *no missing values* in the sentence pairs or label columns. The Hugging Face `datasets` library provides complete tables; we verified that `isnull().sum()` returns zero for all splits. Consequently, no imputation or row dropping was necessary.


== TF-IDF Representation

Classical TF-IDF (Term Frequency-Inverse Document Frequency) represents text as sparse vectors where each dimension corresponds to a unique term (unigram) in the corpus. We use the implementation from `sklearn.feature_extraction.text.TfidfVectorizer` with its *default parameters*: unigram range `(1,1)`, `use_idf=True`, and `smooth_idf=True` (adds one to document frequencies to avoid division by zero). We intentionally avoid n-grams or character-level features because our goal is to establish a *minimal lexical baseline*; adding n-grams would artificially inflate performance on datasets like MRPC where short phrasal overlap is high, obscuring the gap between classical and neural methods.

The vectorizer is fitted on a *combined corpus* comprising all `sentence1` and `sentence2` strings from training splits of all five datasets (STS-B, SICK, STS-Annual, MRPC, and a 5k sample of QQP). This ensures that the TF-IDF vocabulary includes domain-specific terms from every benchmark, preventing out-of-vocabulary words during evaluation. The resulting vocabulary size is approximately 95k terms (depending on the exact training sample), which is highly *sparse*—each sentence is represented as a vector with only 5-30 non-zero entries. This sparsity is an *interpretability advantage*: one can directly inspect which terms drive similarity (e.g., overlapping content words) and diagnose failures (e.g., synonym substitution yields zero overlap). Cosine similarity between two such vectors reduces to the dot product over their intersecting non-zero dimensions, which is computationally efficient and naturally handles variable-length inputs without padding.

== Word2Vec Mean-Pooling Representation

We train a Word2Vec model from scratch using the `gensim` library on the *same combined training corpus* used for TF-IDF (all sentence pairs from all training splits). The model uses the Skip-gram architecture (`sg=1` by default in gensim 4.x with `Word2Vec`), a vector dimensionality of *100*, a context window of 5, and a minimum count of 1 (`min_count=1`) to retain rare words that may carry semantic signal in small benchmarks. The 100-dimensional embedding size represents a *trade-off*: lower dimensions (e.g., 50) reduce expressivity and fail to capture fine-grained synonymy; higher dimensions (e.g., 300) increase memory and training time without meaningful gain on sentence-level tasks given our modest corpus size (\~50k sentences). We set `workers=4` for parallelisation and fix the random seed for reproducibility.

Sentence representations are constructed via *mean pooling* of the word vectors. We choose mean pooling over max pooling or concatenation because: (i) mean pooling is the most common baseline in early STS literature, enabling direct comparison with published results; (ii) max pooling tends to overemphasise salient words (e.g., negations) while ignoring grammatical structure, which harms performance on contradiction detection; (iii) concatenation of mean and max vectors would increase dimensionality to 200 without clear evidence of benefit for static embeddings. We *do not apply inverse document frequency (IDF) weighting* to the word vectors. While IDF weighting can downweight common function words, preliminary experiments (not reported here) showed negligible gains on STS-B ($Delta r < 0.01$) at the cost of additional computation. This remains a limitation: a systematic study of weighting schemes (e.g., smooth inverse frequency, Arora et al. 2017) could potentially improve Word2Vec's surprisingly low STS-B correlation ($r=0.109$).

=== Threshold Optimisation for Classification

For binary classification datasets (MRPC and QQP), each model produces continuous similarity scores. TF-IDF, Word2Vec, and SBERT return cosine similarity values ranging from -1 to +1, while BERTScore returns F1 values between 0 and 1. To convert these continuous outputs into binary predictions (paraphrase/duplicate or not), we need a decision threshold tau. A pair is classified as positive (paraphrase or duplicate) if its similarity score meets or exceeds tau; otherwise, it is classified as negative.

Rather than assuming tau equals 0.5 (a common but often suboptimal default), we tune tau on the training split of each dataset using a simple grid search. The search range spans tau values from 0.10 to 0.90 in increments of 0.05. We exclude values below 0.10 because they would classify nearly all pairs as positive, and values above 0.90 because they would classify nearly all pairs as negative. Both extremes produce trivial classifiers with no practical utility.

For each candidate tau, we compute the macro-averaged F1-score on the training set and select the tau that yields the highest F1. We choose F1 over accuracy because both MRPC and QQP exhibit class imbalance. In MRPC, only about 32 percent of validation examples are true paraphrases (positive class); in QQP, roughly 37 percent are duplicates. Accuracy would favour a trivial strategy of always predicting the majority class, which achieves 68 percent on MRPC but fails completely at identifying paraphrases. F1 balances precision and recall, providing a more meaningful measure for downstream applications where false positives and false negatives may carry different costs (for example, flagging a non-duplicate question as duplicate may annoy users, while missing a true duplicate may degrade search quality).

After selecting the optimal threshold, we evaluate the model on the held-out test set using both accuracy and F1, as reported in the Results section. This protocol ensures that the threshold is chosen without peeking at test data, preserving the validity of our generalisation estimates. The same grid search procedure is applied independently for each model and each dataset, because optimal thresholds vary considerably. For instance, fine-tuned SBERT on MRPC requires a relatively high threshold around 0.72, while TF-IDF on QQP performs best with a much lower threshold around 0.55.

== Sentence-BERT (SBERT)

We evaluate two variants of Sentence-BERT@reimers-2019-sentence-bert: (1) a pretrained model fine-tuned on natural language inference (NLI) data, and (2) a task-specific variant fine-tuned directly on STS-B using cosine similarity loss. Both utilize a siamese transformer architecture that encodes sentences independently before computing cosine similarity. The fine-tuned variant is optimized via a DataLoader with batched training, allowing the model to adapt its representation space to the continuous similarity distribution of the target benchmark.

== Similarity Computation

For vector-based methods (TF-IDF, Word2Vec, SBERT), similarity is computed as the cosine similarity between encoded sentence representations. BERTScore is evaluated as a token-level neural reference metric, leveraging dynamic alignment over contextualized token embeddings without requiring sentence-level pooling. All continuous similarity scores are normalized to $[0,1]$ where applicable (e.g., STS-B raw scores divided by 5.0) to ensure metric comparability across datasets.

= Datasets

- STS-B @huggingface:dataset:stsb_multi_mt — continuous human similarity ratings
- SICK-R @marelli-etal-2014-sick — relatedness scores + entailment labels
- STS-Annual @huggingface:dataset:stsb_multi_mt — multi-domain SemEval shared-task pairs
- MRPC @wang2019glue — binary paraphrase identification (news)
- QQP @wang2019glue — large-scale duplicate question detection

STS-B, SICK-R, and STS-Annual are evaluated as regression tasks using Pearson correlation ($r$) and Spearman rank correlation ($p$). MRPC and QQP are treated as binary classification tasks using Accuracy and F1-score. Continuous scores are normalized to $[0,1]$ prior to thresholding. STS-B and SICK are regression benchmarks (Pearson r, Spearman p). STS-Annual extends the regression suite across six heterogeneous domains (headlines, forum answers, image captions, Twitter, student answers). MRPC and QQP are binary classification benchmarks (Accuracy, F1). A fifth method, BERTScore, is included as a token-level neural reference metric. Statistical significance between model pairs is assessed via Fisher z-transformation.

= Experimental Setup

All experiments are conducted in a unified Python pipeline. Continuous similarity scores are evaluated directly against human annotations using Pearson and Spearman correlations. For classification benchmarks (MRPC, QQP), an optimal decision threshold is determined via grid search on the training split, and evaluation metrics are computed on the held-out test set. Model rankings are generated by sorting on F1-score, with results tabulated for reproducibility.

Statistical significance between model pairs on regression benchmarks is assessed using the Fisher z-transformation on Pearson correlation coefficients, with significance reported at $p < 0.05$. Visualization of cross-dataset performance is generated via multi-panel bar charts comparing regression and classification metrics across all five benchmarks.

= Results

== STS-B & Semantic Similarity Benchmarks (STS-B, SICK, STS-Annual)

On the STS-B regression benchmark, transformer-based methods consistently outperform classical and static embedding baselines.
TF-IDF achieves a Pearson $r$ of $0.690$, reflecting its reliance on surface lexical overlap.
Word2Vec performs poorly on STS-B ($r=0.109$), indicating that static embeddings fail to capture fine-grained semantic similarity when lexical variation is high.
The pretrained SBERT model reaches $r=0.838$, while the STS-B fine-tuned SBERT variant achieves $r=0.868$, demonstrating the benefit of task-specific optimisation.
BERTScore underperforms on STS-B ($r=0.569$) compared to SBERT, despite its token-level alignment mechanism.

Similar trends hold for SICK and STS-Annual.
On SICK, fine-tuned SBERT achieves r=$0.860$ (pretrained: $0.857$), Word2Vec reaches only $0.385$, and BERTScore reaches $0.655$.

On STS-Annual, fine-tuned SBERT again leads with $r=0.868$, while Word2Vec collapses to $0.109$, the domain shift penalises static embeddings severely.

== Classification Benchmarks (MRPC & QQP)

Threshold optimisation on the training splits yields strong classification performance for contextual models.

On MRPC, fine-tuned SBERT achieves an F1-score of $0.837$ (optimal threshold $~0.72$), outperforming pretrained SBERT ($0.832$) and Word2Vec ($0.812$).
Notably, TF-IDF performs competitively on MRPC ($F 1=0.821$) due to high lexical overlap in the dataset.
BERTScore reaches $F 1=0.822$, slightly below fine-tuned SBERT.

On QQP, the gap between classical and transformer methods widens due to the dataset's scale and lexical variation.
Fine-tuned SBERT reaches $F 1=0.739$, while pretrained SBERT achieves $0.737$.
TF-IDF drops to $0.641$ and Word2Vec to $0.559$, confirming that lexical and static methods cannot handle paraphrastic variance in question matching.
Accuracy trends mirror $F 1$.

== Cross-Dataset Performance

The aggregated performance metrics across all five benchmarks are summarized in the following heatmap, which visualizes the Pearson correlation coefficients ($r$) for regression tasks and $F 1$-scores for classification tasks.

#figure(
  image("data/heatmap_all_datasets.png")
)

The heatmap reveals a distinct stratification in the representational capabilities of the evaluated methods. The fine-tuned SBERT model consistently dominates across all domains, maintaining high similarity scores even when facing domain shifts in STS-Annual. Conversely, Word2Vec exhibits severe performance collapse on regression benchmarks (STS-B and STS-Annual), where its static nature fails to capture contextual nuances, resulting in correlations as low as $r=0.109$. 

Interestingly, TF-IDF outperforms Word2Vec significantly on regression tasks but remains competitive with deep learning models on lexical-overlap-heavy classification tasks like MRPC.

A notable anomaly appears in the BERTScore metrics. Despite its sophisticated token-level alignment mechanism, it consistently underperforms compared to both classical TF-IDF and static Word2Vec across all five benchmarks (ranging from $r=0.569$ on regression to $F 1=0.651$ on QQP). This is particularly striking given that SBERT variants achieve F1-scores exceeding $0.83$ on the same classification datasets.


== Discussion of Results

The results delineate a clear hierarchy in semantic representation fidelity, moving from surface-level statistics to deep contextual understanding. The primary driver of performance improvement across all benchmarks is the transition from static to contextual embeddings. While TF-IDF leverages lexical overlap, effectively explaining its resilience on MRPC where paraphrasing often relies on synonym substitution, it fundamentally lacks the ability to resolve polysemy or capture syntactic reordering, leading to poor generalization on datasets with high semantic variance like STS-B.

The failure of Word2Vec on regression benchmarks highlights the critical limitation of averaging static embeddings: it assumes that sentence meaning is a simple sum of word meanings. This compositional fallacy renders the model incapable of distinguishing between sentences with identical bag-of-words but different truth conditions, resulting in the observed low correlations. In contrast, transformer based architectures excel by encoding local and global context, allowing the representation to adapt dynamically to the specific semantics of each input pair.

The fine-tuning process on SBERT proves to be a decisive factor in maximizing performance. The gap between pretrained and fine-tuned SBERT, though modest (e.g., $Delta r=0.03$ on STS-B), is consistent across all tasks. This confirms that adapting the pooling mechanism or embedding space to the target distribution significantly enhances discriminative power, pushing the model closer to human similarity judgments.

The unexpected underperformance of BERTScore warrants specific attention. Although it was designed as a state-of-the-art metric for generation evaluation, its application here as a sentence similarity estimator yields suboptimal results compared to even simple term-frequency baselines. This suggests that the greedy alignment mechanism inherent in BERTScore prioritizes token-to-token correspondence over global semantic equivalence. Consequently, it produces low-variance scores on clean, short-sentence datasets where human annotators penalize minor lexical mismatches heavily, regardless of overall meaning preservation. This result illustrates a crucial domain mismatch: strong performance on generation adequacy does not automatically transfer to similarity ranking tasks without architectural adjustments (e.g., pooling strategies) specifically targeted at sentence-level semantics.

= Error Analysis

The evaluation on the main benchmarks reveal aggregate trends, but the analysis of specific failure modes categorized by lexical distance, syntactic structure, and semantic polarity exposes the distinct blind spots of each representational family. @sentence-similarity-table below details performance scores across carefully constructed test cases that isolate these critical dimensions.

#set page(columns: 1)

#let results = csv("data/sentence_similarity_results.csv",
          row-type: array)

#figure(
  table(
    columns: 8,
    align: center,

    ..results.flatten()
  )
  ,
  caption: "Hand crafted test cases and results"
) <sentence-similarity-table>
#set page(columns: 2)

The qualitative error analysis reveals systematic failure patterns across representational families. TF-IDF collapses catastrophically on lexical paraphrase (cat/feline, sleeping/dozing), achieving near-zero cosine similarity (0.2173) despite perfect meaning preservation, because its bag-of-words representation treats synonymy as surface mismatch. This explains TF-IDF's paradoxical performance: it excels on MRPC where paraphrases share high lexical overlap but fails on STS-B where annotators reward synonym substitution.

Word2Vec demonstrates surprising resilience on lexical variation ($0.9825$ on case [A]), capturing synonymy through distributional neighborhoods, yet fails dramatically on negation (case [D]: $0.8529$ vs. gold low similarity) and attachment ambiguity (case [C]: $0.8789$ vs. gold $0.92$). The negation failure is particularly telling — Word2Vec's additive composition cannot represent logical cancellation, treating "did not recover" as a variant of "recovered."

The fine-tuned SBERT variant shows the most sophisticated behavior on ambiguous constructions, correctly assigning $0.9227$ to the PP-attachment pair where TF-IDF ($0.6067$) and Word2Vec ($0.8789$) are uncertain. However, fine-tuning introduces a vulnerability: on case [D] (negation), fine-tuned SBERT ($0.4916$) underperforms even TF-IDF ($0.5422$), suggesting that the STS-B training distribution (where most pairs have positive similarity) biases the model away from true contradiction detection.

Most striking is the domain-register failure (case [E]): all models except Word2Vec perform poorly on the financial paraphrase ("stock market fell sharply" vs. "equity markets recorded losses"). Word2Vec's $0.7995$ exceeds both SBERT variants ($0.6553$, $0.5191$) and TF-IDF ($0.0$), suggesting that static embeddings preserve domain synonymy relationships that contextual models may overfit to surface phrasing.

= Discussion

== Summary of Principal Findings

This study provides a systematic comparative evaluation of five semantic similarity methods across five complementary benchmarks, yielding four principal findings.

First, transformer-based methods consistently outperform classical and static embedding baselines across all evaluation settings. The fine-tuned SBERT variant achieves state-of-the-art performance on regression tasks (STS-B: $r=0.868$, SICK: $r=0.860$) and classification benchmarks (MRPC $F 1=0.837$, QQP $F 1=0.739$). The performance gap widens as task complexity increases: on STS-Annual, which spans six heterogeneous domains (headlines, forums, image captions, Twitter, answers), fine-tuned SBERT maintains $r=0.868$ while Word2Vec collapses to $r=0.109$. This robustness to domain shift represents the clearest advantage of deep contextual representations.

Second, the transition from static to contextual embeddings is the primary driver of performance improvement, but fine-tuning provides additional, statistically significant gains. On STS-B, the gap between pretrained SBERT ($r=0.838$) and fine-tuned SBERT ($r=0.868$) yields $Delta r=0.03$. Fisher z-transformation confirms significance at $p<0.05$ $(z=4.87, *)$. This consistent advantage across all five benchmarks confirms that task-specific adaptation of the pooling mechanism and embedding space meaningfully aligns representations with human similarity judgments.

Third, classical TF-IDF remains surprisingly competitive on classification tasks where paraphrases exhibit high lexical overlap. On MRPC, TF-IDF achieves $F 1=0.821$, outperforming Word2Vec ($0.812$) and BERTScore ($0.822$), and trailing fine-tuned SBERT by only $0.016$. However, on QQP — where lexical variation is substantially higher — TF-IDF drops to $0.641$, a $0.180$-point gap behind fine-tuned SBERT. This pattern indicates that TF-IDF's utility is bounded by dataset characteristics: it excels when paraphrases rely on synonym substitution within a fixed vocabulary but fails when questions require abstraction across phrasings.

Fourth, BERTScore's systematic underperformance across all five benchmarks ($r=0.569$ on STS-B, $F 1=0.822$ on MRPC, $F 1=0.651$ on QQP) reveals a fundamental task mismatch. Despite its sophisticated token-level alignment mechanism and strong performance on machine translation evaluation (BLEU/ROUGE-style tasks), BERTScore prioritizes token-to-token correspondence over global semantic equivalence. This produces low-variance scores that fail to discriminate between genuine paraphrases and lexical variants. The implication is clear: strong performance on generation adequacy does not transfer to similarity ranking without architectural modifications (pooling strategies, contrastive objectives) specifically designed for sentence-level semantics.

== Interpretation of the Negative Results

The failure of Word2Vec on regression benchmarks warrants careful interpretation. On STS-B, Word2Vec achieves $r=0.109$ — effectively no correlation with human judgments. It represents catastrophic failure of the representational assumption that sentence meaning is a simple sum of word meanings. 

Similarly, the domain-register failure reveals that Word2Vec's preservation of synonymy relationships (stock market ↔ equity markets) gives it an advantage over contextual models on this specific probe. However, this narrow advantage does not translate to benchmark-level performance because real-world similarity judgments require more than lexical synonymy.

The fine-tuned SBERT's vulnerability to negation ($0.4916$ on case [D]) suggests an important limitation of training on STS-B, where the score distribution is skewed toward positive similarity. This observation motivates future work on balanced training that explicitly includes negative examples and contradiction pairs.

== When to Choose Which Model — A Practical Guide

The empirical results across five benchmarks reveal not only a performance hierarchy but also distinct trade-offs that should guide model selection in real-world NLP applications. Below we synthesise our findings into actionable recommendations organised by deployment constraints and task characteristics.

=== Choose TF-IDF When Lexical Overlap Dominates

TF-IDF remains a surprisingly strong baseline for tasks where paraphrases largely preserve surface vocabulary. On MRPC, TF-IDF achieves an F1 of 0.821, trailing fine-tuned SBERT by only 0.016 points. This makes TF-IDF the preferred choice under three conditions: (i) CPU-only environments where transformer inference latency is prohibitive (TF-IDF processes thousands of pairs per second on a single core); (ii) applications requiring interpretability, such as legal discovery or plagiarism detection, where the specific overlapping terms that drove a similarity judgment must be auditable; and (iii) domains with limited vocabulary shift, such as technical document comparison or version control diff analysis. However, TF-IDF should be avoided when paraphrases rely heavily on synonym substitution (e.g., "car" vs. "automobile") or when syntactic reordering is common, as demonstrated by its poor performance on STS-B (Pearson r = 0.690) compared to SBERT (r = 0.868).

=== Choose Word2Vec Only Under Extreme Resource Constraints

Word2Vec's performance on regression benchmarks is catastrophic (STS-B r = 0.109), and even on classification tasks it underperforms TF-IDF in most settings. The only scenarios where Word2Vec may be justified are those with extremely limited computational budgets (e.g., edge devices with \<50MB memory) where even TF-IDF's vocabulary storage is too large, or highly specialised domains where synonymy relationships are stable and contextual nuance is minimal. One positive signal from our qualitative probes is Word2Vec's strong performance on the financial domain paraphrase (case E, similarity = 0.7995), where it outperformed both SBERT variants. This suggests that static embeddings may preserve domain-specific synonym clusters that contextual models sometimes overfit. However, given the unreliability of Word2Vec across broader evaluations, we recommend exhaustive validation before deployment.

=== Choose Pretrained SBERT for General-Purpose Similarity

The off-the-shelf `all-MiniLM-L12-v2` model delivers strong performance across all five benchmarks without any fine-tuning. It achieves a Pearson correlation of 0.838 on STS-B, 0.857 on SICK, and F1 scores of 0.832 on MRPC and 0.737 on QQP. This model is the best choice for most applied settings where: (i) no labelled in-domain data exists for fine-tuning; (ii) the task is general-purpose semantic similarity (e.g., search ranking, clustering, duplicate detection across mixed domains); (iii) computational resources are modest (the model runs on CPU at \~50ms per pair or GPU at \~5ms per pair). Pretrained SBERT offers the best accuracy-to-effort ratio and serves as a robust default baseline for any new STS task.

=== Choose Fine-Tuned SBERT When Maximum Accuracy Is Required

Fine-tuning SBERT on task-specific data yields consistent, statistically significant improvements across all benchmarks. On STS-B, fine-tuning increases Pearson r from 0.838 to 0.868 (Δ = 0.03, p < 0.05). On MRPC, F1 improves from 0.832 to 0.837. While these absolute gains appear modest, they represent meaningful reductions in prediction error, particularly in high-stakes applications such as duplicate bug report detection in software engineering or citation recommendation in academic search. Fine-tuning is recommended when: (i) labelled in-domain data is available (even a few thousand examples suffice, as shown by our 5k STS-B sample); (ii) the target domain matches the fine-tuning distribution (STS-B fine-tuning transfers well to SICK and STS-Annual but may degrade on negation cases, as shown in our error analysis); (iii) the cost of computation (1-2 GPU hours for models of this size) is acceptable relative to the performance gain.

=== Avoid BERTScore for Similarity Ranking

BERTScore consistently underperformed expectations across all five benchmarks, achieving Pearson correlations of only 0.569 on STS-B and 0.655 on SICK, and F1 scores of 0.822 on MRPC and 0.651 on QQP. This is particularly striking given that BERTScore uses the same underlying transformer architecture (RoBERTa-large) as SBERT variants. The explanation lies in task mismatch: BERTScore was designed for machine translation evaluation, where it greedily aligns tokens between a candidate and a single reference translation. This token-level alignment prioritises lexical fidelity over global semantic equivalence. In similarity ranking tasks, two sentences can be semantically equivalent without token-level overlap (e.g., "the cat slept" vs. "the feline dozed"). BERTScore penalises such pairs, producing low-variance scores that fail to discriminate along the full similarity continuum. For similarity ranking, we strongly recommend SBERT or fine-tuned SBERT over BERTScore. Only consider BERTScore if your task is explicitly generation evaluation (e.g., summarisation, translation) where token coverage matters.


= Conclusion and Future Work

This paper presented a comprehensive comparative evaluation of five semantic similarity methods spanning classical statistical representations (TF-IDF), static distributed embeddings (Word2Vec), and transformer-based contextual representations (pretrained and fine-tuned Sentence-BERT), with BERTScore included as a token-level neural reference metric.

Evaluation across three regression benchmarks (STS-B, SICK-R, STS-Annual) and two binary classification tasks (MRPC, QQP) establishes a clear hierarchy: fine-tuned SBERT > pretrained SBERT > BERTScore ≈ TF-IDF > Word2Vec.

The principal contributions are: (i) a unified experimental pipeline enabling fair comparison under identical preprocessing, scoring, and thresholding conditions; (ii) statistical significance testing via Fisher z-transformation confirming the advantage of fine-tuning over pretrained representations; (iii) qualitative error analysis identifying systematic failure modes including negation blindness, domain-register mismatch, and lexical variation sensitivity; and (iv) the surprising finding that BERTScore, despite strong performance on generation evaluation, underperforms simple term-frequency baselines on similarity ranking tasks.

The results have clear practical implications: for applications requiring fine-grained similarity ranking (e.g., duplicate detection, information retrieval, conversational AI), fine-tuned Sentence-BERT provides the best accuracy, though the marginal gain over pretrained SBERT ($Delta r=0.03$) may not justify fine-tuning costs in resource-constrained settings. For classification tasks with high lexical overlap, TF-IDF remains a strong, interpretable baseline.

Future work should investigate three directions. First, training on heterogeneous NLI and STS data combined may improve generalization across linguistic phenomena, particularly negation and logical contradictions.

Second, evaluation of recent large language models (GPT, LLaMA) as similarity estimators using prompting or embedding extraction could establish whether scale alone resolves the limitations identified here.

== Limitations of This Study

While our experimental framework provides a systematic comparison across five benchmarks and five model families, several important limitations should be acknowledged. These constraints do not invalidate our findings but rather delineate the boundaries within which our conclusions generalise.

=== Monolingual Scope (English Only)

All datasets used in this evaluation are exclusively English. STS-B, SICK, MRPC, QQP, and STS-Annual contain only English sentence pairs. Consequently, our findings regarding the superiority of fine-tuned SBERT over classical methods may not transfer to other languages, particularly those with rich morphology (e.g., Finnish, Turkish), non-concatenative morphology (e.g., Arabic), or limited pretrained transformer availability. For low-resource languages where multilingual models like mBERT or XLM-R are required, the performance gap between static and contextual embeddings may differ substantially. Future evaluations should replicate this framework on multilingual STS benchmarks such as STS-2017 (Arabic, Spanish) or the XLING-SemEval datasets.

=== Fine-Tuning Confined to a Single Dataset

We fine-tuned SBERT exclusively on STS-B (using a 5,000-sample subset). We did not explore fine-tuning on SICK, MRPC, or QQP separately, nor did we investigate multi-task or cross-dataset fine-tuning (e.g., training on STS-B + MRPC jointly). This design choice was intentional: we wanted to isolate the effect of fine-tuning on a canonical similarity regression task. However, it also means that our fine-tuned model is specialised for continuous similarity prediction and may not be optimal for paraphrase detection (MRPC) or duplicate question identification (QQP). A more comprehensive study would compare task-specific fine-tuning (e.g., fine-tune separately on each benchmark) versus unified fine-tuning on a mixture of STS and NLI data. The latter approach, popularised by Sentence-BERT's original paper, often yields better generalisation across diverse tasks.

=== Simple Grid Search for Threshold Optimisation

For binary classification tasks (MRPC, QQP), we selected decision thresholds using a grid search over tau from 0.10 to 0.90 in steps of 0.05. While this method is transparent and computationally cheap, it is also coarse. The optimal threshold for a given model and dataset may fall between grid points (e.g., 0.73, which we would round to 0.70 or 0.75). More sophisticated methods exist, including Youden's index (maximising sensitivity + specificity) derived from the receiver operating characteristic curve, or cost-sensitive thresholding that accounts for asymmetric misclassification costs (e.g., false negatives may be more expensive than false positives in duplicate detection). Our simple grid search likely underestimates the true achievable F1 for all models, though the relative rankings are probably stable because the same coarse grid applies uniformly across models.

=== Potential Data Contamination

The pretrained SBERT model (`all-MiniLM-L12-v2`) was trained on a large corpus of NLI and STS data that may include examples from our test sets. While the original Sentence-BERT paper carefully removed benchmark overlaps, the exact training data for the `all-MiniLM` variant is not fully documented. There is a risk of data contamination (i.e., the model has already seen some STS-B or MRPC examples during pretraining), which would artificially inflate its zero-shot performance. Our fine-tuned variant, trained only on a 5,000-sample subset of STS-B training data, is less susceptible to this concern because the fine-tuning split is disjoint from the test set. Future work should verify contamination by training models from scratch on explicitly non-overlapping corpora