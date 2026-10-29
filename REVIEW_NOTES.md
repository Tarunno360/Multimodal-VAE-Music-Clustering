# Review notes

This repository preserves the reviewed study's code, numeric results, and PDF
reports. Model source files and frozen experiment configurations are unchanged.

The extension called `direct_spectral` uses flattened pooled spectrograms with
the common clustering evaluator. It is a direct spectrogram-feature baseline,
not the spectral clustering algorithm. Original result keys and reports are kept
unchanged to avoid silently relabeling historical evidence.

The public snapshot omits trained checkpoints and fitted transformations. Saved
verification logs describe the original workspace. Regenerate the experiments
before running checkpoint-verification scripts on a fresh clone.

The original data source, rights, and preprocessing limitations still apply.
Repository availability does not establish recording provenance or permission.
