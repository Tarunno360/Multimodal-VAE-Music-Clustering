# Neural Networks project: VAE for hybrid-language music clustering

Technical brief supplied by the user in this chat on 29 October 2026.
Prepared by Moin Mostakim. Original course due date: 10 January 2026.
This transcription preserves the substantive requirements; the supplied NeurIPS
files describe formatting only.

## Deliverables

1. GitHub repository with implementation, dataset processing scripts and results.
2. NeurIPS-like scientific PDF covering method, experiments, results and discussion.

## Easy

- Basic VAE for music feature extraction.
- Small mixed English + Bangla song dataset.
- K-means on latent features; t-SNE or UMAP visualization.
- Compare PCA + K-means using silhouette and Calinski--Harabasz index.

## Medium

- Convolutional VAE for spectrograms or MFCCs.
- Hybrid audio + lyric representations.
- K-means, agglomerative clustering and DBSCAN.
- Silhouette, Davies--Bouldin and ARI when labels exist.
- Explain why representations perform better or worse than baselines.

## Hard

- Conditional VAE or beta-VAE; investigate disentangled representations.
- Multimodal clustering combining audio, lyrics and genre information.
- Silhouette, NMI, ARI and cluster purity.
- Latent plots, cluster distributions over languages/genres, reconstructions.
- PCA + K-means, autoencoder + K-means and direct spectral-feature clustering.

Conditioning on genre is permitted for this task but must be identified as using
labels; it cannot be presented as a fully unsupervised comparison. A beta-VAE
architecture alone is not evidence of disentanglement.

## Metrics

- Silhouette: average `(b(i)-a(i))/max(a(i),b(i))`; larger is better.
- Calinski--Harabasz: `[tr(B)/(k-1)]/[tr(W)/(n-k)]`; larger is better.
- Davies--Bouldin: average across clusters of the largest ratio of combined
  within-cluster scatter to centroid distance; smaller is better.
- ARI: chance-adjusted Rand agreement; larger is better. Zero is the expected
  chance level, not a guarantee that a specific clustering is random.
- NMI: `2 I(U;V)/(H(U)+H(V))`; larger is better.
- Purity: `sum_k max_j |cluster_k intersect class_j| / n`; larger is better,
  but can increase trivially with the number of clusters.

## Mark allocations and judging

Easy 20; Medium 25; Hard 25; metrics 10; visualizations 10; report 10; repository
10. These supplied allocations total **110**, so no normalization or assumed
grade is applied here. Judging emphasizes correctness, multimodal features,
clustering quality, clear plots, baseline comparisons, report clarity,
reproducibility, and justified creative extensions.

Suggested organization: data, notebooks, source modules, results, README and
requirements. Suggested report sections: Abstract, Introduction, Related Work,
Method, Experiments, Results, Discussion, Conclusion, References.

The user will choose a conference after the work is finished; no venue or
current submission deadline is fixed.
