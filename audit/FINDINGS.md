# Audit of the supplied project

Audit date: 29 October 2026 (Asia/Dhaka). Original numbered folders are preserved.
The original paper sources and PDF are backed up in `audit/original_writing/`.

## Requirements available

`Project Requirement/neurips_2024.tex` and its PDF are the unmodified NeurIPS
2024 formatting example, not a technical course assignment. They specify a
nine-page main text, consistent references, legible figures, and a research
checklist. The technical rubric was initially absent, then supplied by the user in this
chat and saved as `Project Requirement/assignment_brief.md`. The main experiment
was already frozen; a documented extension adds direct spectral clustering and
a multimodal genre oracle. A new bilingual course demonstration supplies the
missing English--Bangla input coverage. The 2024 template does not establish
current conference compliance.

## Evidence and corrective actions

| Finding | Source evidence | Consequence / repair |
|---|---|---|
| 796 metadata rows, 683 processed songs | `3/data/dataset.csv`; `2/data/processed_ids.npy` | Join metadata by the saved ID order, never by array position or un-ordered membership filtering. IDs are unique, all resolve, and this particular archive's filtered CSV order happens to match. |
| Genre labels exist | Metadata has `country`, `pop`, `rap`, `rb`, `rock` | Replace claims that ground-truth genres are absent. Reserve labels for external evaluation, except a separately identified conditional oracle. |
| Cached audio has no raw source in the delivery | None of the 683 referenced audio paths exists | Experiments reproduce from supplied arrays only. Sample rate, window, mel parameters, clip origin, and normalization cannot be independently verified. |
| Multilingual claim unsupported | No language column, no Bengali-script lyrics among the processed rows | Reframe as audio--lyrics music clustering. Absence of Bengali script is not a language detector and does not prove all lyrics are English. |
| Transformer identity and row provenance unrecorded | `1/data/lyrics_embeddings.npy` is 683 x 384, but no saved extraction manifest or embedding ID sidecar | Do not identify it as BERT or assume dimension establishes provenance. Use text features rebuilt from ID-aligned lyrics for primary experiments. |
| Synthetic experiment mixed with real project | `3/notebooks/VAE_crrpt.ipynb`, cell 5 generates 500 synthetic examples with artificial genre/language offsets | Exclude from evidence about real songs. Keep as legacy demonstration only. |
| Failed notebook execution | `1/notebooks/VAE_crpt3_Task.ipynb`, cells 9 and 20 contain 796-versus-683 sample errors | Replace the active execution path with an ID-validated package and integration smoke run. |
| Shuffled feature/label order risk | Same PyTorch notebook extracts latent features using the shuffled training loader | Extract in deterministic original row order; save predictions with ID and split manifests. |
| Decoder range mismatch | Hard-task notebook standardizes audio/text, then uses sigmoid decoders; medium text decoder also uses sigmoid on standardized text | Use unrestricted linear outputs and squared reconstruction loss for signed continuous targets. |
| KL/reconstruction scaling ambiguous | Hard task combines mean reconstruction loss and sum-over-latents KL | Specify reductions and coefficients explicitly, sweep regularization, log KL and active units. Do not assume a numeric beta matches another implementation. |
| Conditional VAE given evaluated labels | Hard task concatenates genre one-hot vectors during training and inference | Conditional results are an oracle diagnostic, not an unsupervised competitor. |
| Preprocessing fitted globally | Scalers/PCA in legacy notebooks are fitted before the split | Fit vocabulary, SVD, scaling and PCA on training rows only; freeze test transformations. |
| Most clustering results reuse training examples | Latents extracted on the entire dataset; clustering fit and scored there | Separate inductive K-means on held-out tracks from transductive Ward/DBSCAN descriptions. |
| Spectral baseline duplicates PCA | Hard task creates both with PCA on the same flattened standardized arrays | Remove the duplicate; use explicitly defined time-mean/std spectrogram features as a separate baseline. |
| Incorrect noise handling comments | Hard task's evaluation passes -1 to sklearn metrics while claiming sklearn ignores it | Explicitly exclude noise and report coverage; undefined cases remain null, not invented zeros. |
| Degenerate partitions called success | Legacy beta-VAE K-means sizes [679,1,1,1,1], silhouette 0.894; conditional sizes [678,1,1,1,2], silhouette 0.954 | Report semantic agreement, largest-cluster share, effective cluster count, and KL diagnostics alongside geometry. Cluster imbalance alone does not establish posterior collapse. |
| Contradictory conclusions | Discussion says VAEs always beat linear baselines although the hard-task table favors baseline NMI/ARI | Write conclusions only after the new measured comparison; distinguish geometry from genre agreement. |
| Unsupported semantic interpretation | Draft assigns genre meanings to clusters without an ID-level analysis or listening study | Remove speculative genre/mood descriptions. Provide actual cross-tabulations. |
| Incomplete bibliography | `mcinnes2018umap` cited but absent; duplicated bibliography-style declarations | Repair references and compile with no unresolved citations. |
| Unreproducible paths and model exports | Notebooks use `/content/drive/...`; only disconnected H5 files are provided | Provide local relative paths, configuration files, saved fitted transformations, model checkpoints, hashes, versions and exact commands. |

## Rebuilt experimental protocol (defined before inspecting new test scores)

- One fixed grouped train/validation/test split, approximately 70/15/15; seed 2026.
- Group exact or normalized duplicate titles, exact normalized lyrics, or identical
  spectrogram arrays. Artist-disjoint evaluation cannot be claimed without artists.
- Fixed 4 x 4 average pooling maps 128 x 256 supplied arrays to 32 x 64. This is
  a computational simplification, not a reproduction of the original architecture.
- Train-only TF-IDF and 64-dimensional truncated SVD create verifiable text features.
- Five training/clustering seeds; 32-dimensional embeddings; fixed k=5, with
  k in {2,3,4,6,8,10} reported as sensitivity analyses, not selected on test labels.
- Audio PCA, text LSA, balanced concatenation PCA, audio-statistics PCA, AE,
  regularization sweep, convolutional VAE, dense multimodal VAE, convolutional
  hybrid VAE, free-bits ablation and shuffled-text negative control.
- Genre-conditional VAE only in a separately labeled oracle analysis.
- Up to 100 epochs; 20-epoch KL warmup; validation-objective checkpointing with
  patience 15 after warmup. Four fixed Monte Carlo samples for validation.
- Train K-means on training embeddings and predict held-out embeddings. Ward and
  DBSCAN are explicitly within-test, transductive descriptions. DBSCAN radius
  candidates derive from training distances and are selected without genre labels.

## Unresolved external inputs

The technical assignment rubric is now supplied. The user attributes upstream
data mostly to a GTZAN Kaggle mirror and received cached arrays from a friend;
an ID-level provenance chain remains absent. Raw-audio preprocessing, dataset
rights, embedding extraction provenance and artist/language metadata for the
683-song corpus remain unresolved. No target venue has been selected.
The repaired project can be experimentally complete from cached arrays, but it
cannot establish these missing facts or guarantee conference acceptance.

## Follow-up split sensitivity

After inspecting the primary results, two further grouped splits (2027 and 2028)
repeat six neural architectures and four baselines with unchanged settings and
five seeds. This adds 60 neural runs and 1,600 evaluation records. The primary
split and bootstrap remain unchanged. These are overlapping resamples of one
corpus, not independent dataset replications. Results and validation are in
`results/analysis/split_sensitivity*` and `audit/robustness_verification.json`.
