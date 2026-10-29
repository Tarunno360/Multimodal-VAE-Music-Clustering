# Multimodal VAE Music Clustering

Audio and lyric representation learning for music clustering, with dense,
convolutional, multimodal, and beta variational autoencoders. The study compares
learned representations against audio PCA, audio statistics, text-only LSA,
fusion PCA, and a deterministic autoencoder.

## Results

The main corpus contains 683 songs with aligned cached spectrograms, lyrics,
and five genre labels. Duplicate groups are kept separate across 478 training,
103 validation, and 102 test songs. Preprocessing is fitted on training data only.
K-means is fitted on training representations and predicts held-out assignments.

| Representation | Test ARI (mean ± sample SD, five seeds) |
|---|---:|
| Audio PCA | 0.086 ± 0.000 |
| Audio VAE | 0.096 ± 0.020 |
| Dense multimodal VAE | 0.153 ± 0.096 |
| Convolutional audio–lyrics hybrid | 0.207 ± 0.029 |
| Text-only LSA | 0.227 ± 0.028 |

The hybrid improves over audio alone in this experiment. An advantage over
text-only features is not established. Two additional grouped splits preserve
this mean ordering; all three splits reuse the same corpus.

A separate audio-only feasibility experiment processes 20 English recordings
and 20 Bangla clips. Clustering fits and scores all 40 clips. Language is
confounded with collection source; this is not a held-out bilingual benchmark.

## Repository layout

- `musicvae/`: ID alignment, training-only transformations, models, metrics, training and evaluation.
- `configs/`: primary, smoke, and follow-up split configurations.
- `scripts/`: input mapping, audio acquisition and processing, analysis, verification, and paper generation.
- `tests/`: alignment, leakage, noise handling, objective, and gradient checks.
- `notebooks/Complete_Project.ipynb`: walkthrough of saved results.
- `results/`: primary, extension, bilingual, and split-sensitivity numeric results.
- `Final Writing Latex/`: manuscript sources, generated tables, figures, and PDFs.
- `output/pdf/`: named and anonymous reports.
- `dataset/`: supplied cached inputs and metadata.
- `model and tasks/`: historical notebooks; these are not the active training path.
- `audit/`: historical findings, source snapshots, and recorded verification results.

## Setup and reproduction

Python 3.12 was used for the original runs. The lock file records that environment;
platform-compatible package versions may be needed on another system.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python scripts/import_legacy_inputs.py
LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m pytest -q
LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m musicvae.experiment --config configs/smoke.json --output results/my_smoke
LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m musicvae.experiment --config configs/main.json --output results/my_replication
```

The input mapper verifies hashes before copying repository data into the frozen
numbered input paths used by the experiments. Use a new output directory for a
fresh run; `--resume` requires an identical configuration and source signature.

The repository includes numeric summaries and histories, but not original
checkpoints, fitted transformations, or representation bundles. Verification
scripts that reload these artifacts require regeneration first. Historical
verification logs record checks performed in the original complete workspace;
they are not a claim of a new execution on every clone.

To regenerate the canonical pipeline, first move the supplied `results/` folder
to a separate backup location, then run:

```bash
LOKY_MAX_CPU_COUNT=4 .venv/bin/python -m musicvae.experiment
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/extend_requirements.py
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/build_report.py
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/run_robustness.py
.venv/bin/python scripts/analyze_robustness.py
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/verify_robustness.py
.venv/bin/python scripts/fetch_bilingual.py
NUMBA_CACHE_DIR="$PWD/tmp/numba" .venv/bin/python scripts/prepare_audio.py --manifest data/bilingual/manifest.json --output data/bilingual/prepared
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/bilingual_experiment.py
LOKY_MAX_CPU_COUNT=4 .venv/bin/python scripts/verify_results.py
.venv/bin/python scripts/build_paper.py
```

Paper compilation requires Tectonic or a suitable TeX Live installation.
The bilingual downloader uses pinned source revisions; network access is required.

## Interpretation and limitations

The historical beta-VAE and conditional results illustrate how high silhouette
can coexist with highly imbalanced clusters and near-zero genre agreement.
The rebuilt evaluation therefore reports external agreement, cluster balance,
reconstruction, KL, and latent diagnostics alongside internal metrics.

The extension named `direct_spectral` actually applies the shared clustering
routines to flattened pooled spectrogram values. It does not implement the
spectral clustering algorithm. The historical name is retained in frozen result
files and PDFs to preserve their identity; interpret it as a direct spectrogram
feature baseline. See `REVIEW_NOTES.md`.

Original recording provenance, reuse rights, and raw-audio preprocessing for the
683-song corpus are unresolved. The cached lyric vectors are not used in the new
primary models because their extraction provenance is missing. Artist-disjoint
and independent-dataset evaluations remain future work. Genre-conditioned models
receive labels at inference and are reported separately as oracles.

See `DATA_CARD.md` for data scope and `CONFERENCE_READINESS.md` for research
limitations. The reports are research drafts, not submitted conference papers.
This repository starts from a single source snapshot with a fresh commit history.
