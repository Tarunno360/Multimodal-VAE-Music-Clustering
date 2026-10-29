# Data card and provenance

## Supplied corpus

- 796 metadata rows; 683 processed IDs, all unique and matched by exact ID.
- 683 x 128 x 256 float32 cached spectrograms; range [-1, 1].
- Genres: rock 178, rap 178, pop 153, country 109, rb 65.
- Lyrics are present for every processed row. Language and artist columns are absent.
- 676 duplicate groups after conservative title/lyric/exact-array matching.
- No referenced raw audio file is present. Original sample rate, segment origin,
  spectrogram parameters, and normalization are unverified.
- User-provided source attribution: [GTZAN Kaggle mirror](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data),
  with cached data received from a friend. This is not an ID-level provenance
  chain. The manuscript does not call the 683-row corpus a verified GTZAN subset.
- Cached 384-dimensional text vectors lack an extraction/model/ID manifest.
  They are preserved and hashed, but excluded from primary training.
- Original dataset and lyric licenses are unresolved; no new license is asserted.
- The existing public repository `Tarunno360/Multimodal-VAE-Music-Clustering`, commit
  `9ec90d546a1f96d5a551f4f1259c0336c7b94c81`, contains byte-identical copies
  of all four frozen inputs. `audit/repository_match.json` records the comparison.
  This verifies file identity and availability, not original recording provenance.

## New bilingual course cohort

| Component | Source | Selection |
|---|---|---|
| English | [JamendoLyrics](https://huggingface.co/datasets/jamendolyrics/jamendolyrics) | All 20 explicitly English-labeled tracks; a 3-second excerpt begins at the first annotated lyric line |
| Bangla | [BanglaBeats](https://huggingface.co/datasets/thisisjibon/banglabeats) | 20 released 3-second clips, spaced over numeric file ranks in eight genre folders |

JamendoLyrics revision: `de188c963fd4539bc769b3feb83582e5a9595e36`.
BanglaBeats revision: `f3c4cd1e6172ac1bbdbb8222f3990f53f9ec7f8d`.
Source file names, URLs, offsets, labels, license strings and recording SHA-256
values are saved in `data/bilingual/manifest.json`.

JamendoLyrics identifies per-song Creative Commons license types; its MIT license
explicitly excludes songs and lyrics. Those separate conditions remain applicable.
BanglaBeats' inspected loader leaves its license field empty. Its release also
does not map numbered 3-second clips back to original songs. No song-identity or
redistribution-rights assertion is made for that release.

The dataset source determines the language label in this cohort. Genre vocabularies,
production conditions and excerpt selection also differ by source. Thus language
agreement is not evidence of language recognition or language-invariant genres.
No Bengali lyrics are fabricated, translated into surrogate paired lyrics, or
borrowed from unrelated songs. The bilingual experiment is audio-only.

## Processing

The new raw-audio recipe is explicit: mono 22,050 Hz, 3-second excerpt, 2,048 FFT,
512 hop, 128 mel bins, squared magnitude, maximum-relative dB limited to 80 dB,
linear interpolation to 256 time frames, then [-1, 1] scaling. Common 4 x 4
average pooling produces 32 x 64 features. See the saved preprocessing JSON for
versions and hashes. This recipe must not be retroactively attributed to the
supplied arrays.

All new recordings remain local. The existing repository already contains the
legacy cached inputs; the new review update does not add or replace them.
The release-candidate archive omits audio,
full lyrics, original feature arrays and checkpoints. It includes code, results,
source manifests and acquisition instructions for review. Public release should
resolve the outstanding data provenance and licensing first.
