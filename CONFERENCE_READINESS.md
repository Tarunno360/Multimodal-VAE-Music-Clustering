# Conference readiness assessment

Assessment date: 29 October 2026. The user will choose a venue after the study.

## Current result

The repaired project is a reproducible empirical audit with a working course
implementation. It demonstrates an important failure of interpreting latent
geometry as semantic quality, an improvement of the hybrid over audio alone on
all three grouped splits, while a simple text-only baseline has higher mean ARI
on each split.
It is **not yet a strong general-purpose conference submission**. The architecture
components, free bits and KL warmup are established methods; do not claim them as
novel. More complicated architecture would not by itself resolve the evidence gaps.

## Work still needed before submission

1. Establish the original 683-song corpus's source, rights and raw preprocessing,
   or replace it with a fully documented corpus. Obtain original-song and artist
   identifiers for Bengali recordings. This is the highest-priority dependency.
2. Build a sufficiently sized paired English/Bangla audio--lyrics collection with
   verified language and genre annotations. Avoid making language identical to
   dataset source; include multiple sources per language and overlapping genres.
   The current 40-clip cohort is only a course feasibility demonstration.
3. Extend the completed three-split sensitivity analysis to artist-disjoint
   evaluation and at least one independently sourced dataset. The follow-up
   grouped splits overlap within the same corpus. The primary bootstrap intervals
   remain conditional on one split and do not establish external validity.
4. Add appropriate documented pretrained audio and multilingual text baselines
   after data provenance is fixed. Preserve text-only, audio-only and simple
   fusion controls. Compare matched dimensions and disclose pretraining overlap.
5. Choose a research claim before further test inspection: for example, when
   lexical information dominates small-corpus multimodal music clustering, or
   how normalization and degeneracy diagnostics change method rankings. Test
   that claim on new held-out data rather than optimizing the existing test set.
6. Perform independent author review of code, references, result tables and
   AI-assisted changes. Check the chosen venue's current format, anonymity, AI-use
   policy, data/code requirements, dates and publication conditions.

## Venue context, not a submission decision

ISMIR explicitly includes evaluation, datasets and reproducibility among its
topics and evaluates scientific quality, reusable insights and novelty; this is
a closer topical fit than treating the NeurIPS course template as a venue choice.
Its 2026 full-paper deadline was 27 April 2026, so it is already past at this
assessment date. See the [official 2026 call](https://ismir2026.ismir.net/authors/call-for-papers).
The [ISMIR society](https://ismir.net/) lists ISMIR 2027 for 12--16 September 2027
in London; no submission deadline is inferred from those event dates.

A relevant workshop may be a more proportionate target for a carefully scoped
audit or negative-result study. Selection should follow the stronger data and
replication work, as requested. No deadline, acceptance prospect or readiness
claim is invented. The [2026 ISMIR AI-use policy](https://ismir2026.ismir.net/ai-usage-policy)
also illustrates why author verification of references and AI-assisted work
matters; recheck the actual target venue's policy when chosen.

## Publication state

Code, saved numeric results, and named/anonymous reports are maintained in this
repository. No conference submission is claimed. The anonymous PDF is a formatting
preview; supplementary assets also require anonymity review before submission.
