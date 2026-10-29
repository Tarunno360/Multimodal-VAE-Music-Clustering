# Requirement-to-evidence map

The actual brief is saved in `Project Requirement/assignment_brief.md`.
The listed mark allocations sum to 110; no estimated grade is claimed.

| Requirement | Implemented evidence | Scope / qualification |
|---|---|---|
| Basic VAE | `musicvae/models.py`, dense VAE runs | Signed linear output, explicit KL reductions and validation checkpointing |
| English + Bangla dataset | `data/bilingual/manifest.json`, `results/bilingual/` | 20 English recordings and 20 Bengali clips; source-confounded feasibility cohort; original Bengali song IDs unavailable |
| K-means latent clustering | `musicvae/experiment.py`; bilingual runner | Main study is inductive; bilingual course demonstration is transductive |
| t-SNE/UMAP | `figures/rebuilt_tsne.*`, `figures/bilingual.*` | t-SNE satisfies the either/or requirement; not used to score clusters |
| PCA comparison, silhouette, CH | Main metrics CSV and bilingual metrics CSV | Five seeds; all undefined metric cases retained |
| Convolutional VAE | `conv_vae` and `hybrid_vae` runs | Fixed pooling to 32 x 64 is disclosed |
| Audio + lyrics | Dense multimodal and convolutional hybrid | ID-aligned train-only TF-IDF/SVD lyric features; no unsupported BERT claim |
| K-means, agglomerative, DBSCAN | `metrics.csv` protocol column | Ward and DBSCAN are separate transductive test descriptions; noise coverage is explicit |
| Silhouette, DB, ARI | Main and extension metric files | Includes genre labels already present in metadata |
| Beta-VAE / conditional VAE | Four beta values; audio and multimodal conditional oracles | KL/active-unit diagnostics; no unmeasured disentanglement claim |
| Audio + lyrics + genre | `multimodal_conditional_oracle` | Genre is used at inference, explicitly distinguished from unsupervised models |
| NMI, ARI, purity | Full metrics and paper tables | Purity compared with majority baseline; fixed k and k-sensitivity reported |
| Genre/language cluster distributions | `rebuilt_genre_distribution`, `bilingual` figures | Actual counts, not inferred cluster names |
| Reconstructions and interpolation | `rebuilt_reconstruction`, `rebuilt_interpolation` | Spectrogram-feature reconstructions, not synthesized audio quality claims |
| PCA, AE, direct spectrogram controls | `audio_pca`, `audio_ae`, `direct_spectral` | Direct spectral uses all 2,048 pooled features without PCA |
| Processing scripts | `scripts/prepare_audio.py`, data/model modules | New raw-audio recipe reproduced on bilingual data; original cached recipe is unknown |
| Reproducible code and results | Configs, hashes, locks, tests, logs, predictions, checkpoints | Local complete package, with public-source candidate archive |
| NeurIPS-like report | Report PDFs in `output/pdf/`; LaTeX sources maintained privately | Main text below supplied nine-page limit; additional references, appendix, checklist |
| GitHub repository | Current repository with code, numeric results, and reports | Fresh history; original model artifacts require regeneration |
| Conference research | Audit, controls, five seeds, three grouped splits and uncertainty analysis | Research draft complete; public provenance, stronger bilingual/artist-independent data and external replication remain |

The implemented course experiments are complete within the explicitly stated
data scope. Missing provenance and bilingual original-track metadata are not
silently replaced with guesses.
