"""Load ID-aligned supplied data; learn transformations on the training set only."""
from pathlib import Path
import hashlib
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def align_metadata(metadata, ids):
    if metadata.id.duplicated().any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Track IDs must be unique in metadata and processed arrays")
    missing = set(ids) - set(metadata.id)
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} processed IDs")
    return metadata.set_index("id", drop=False).loc[ids].reset_index(drop=True)


def duplicate_groups(meta, spectra):
    """Union equal normalized titles, lyrics, or exact spectrograms before splitting.

    Title matching is conservative (can group unrelated songs with the same title).
    Artist-disjoint splitting is impossible without artist metadata.
    """
    parent = list(range(len(meta)))
    def root(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in meta.iterrows():
        title = re.sub(r"\W+", "", str(row.title).casefold())
        lyric = re.sub(r"\s+", " ", str(row.lyrics).replace("\\n", " ").casefold()).strip()
        keys = [("title", title), ("lyrics", lyric),
                ("audio", hashlib.sha256(spectra[i].tobytes()).hexdigest())]
        for key in keys:
            if not key[1]:
                continue
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(meta))])


def make_splits(groups, seed=2026):
    indices = np.arange(len(groups))
    trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed).split(indices, groups=groups))
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=seed+1).split(trainval, groups=groups[trainval]))
    return {"train": np.sort(trainval[tr]), "val": np.sort(trainval[va]), "test": np.sort(test)}


def load_data(root):
    root = Path(root)
    paths = {"spectrograms": root/"2/data/processed_spectrograms.npy",
             "ids": root/"2/data/processed_ids.npy", "metadata": root/"3/data/dataset.csv",
             "cached_lyrics": root/"1/data/lyrics_embeddings.npy"}
    ids = np.load(paths["ids"], allow_pickle=False)
    spectra = np.load(paths["spectrograms"], allow_pickle=False)
    meta = align_metadata(pd.read_csv(paths["metadata"]), ids)
    if spectra.shape != (len(ids), 128, 256) or not np.isfinite(spectra).all():
        raise ValueError("Expected finite (N,128,256) supplied spectrograms")
    if meta[["genre", "lyrics", "title"]].isna().any().any():
        raise ValueError("Metadata contains missing fields")
    groups = duplicate_groups(meta, spectra)
    return spectra, meta, groups, {k: {"path": str(p.relative_to(root)), "sha256": sha256(p)} for k,p in paths.items()}


def prepare_features(spectra, meta, splits, text_dim=64):
    tr = splits["train"]
    # Fixed local average pooling: no parameters are fitted on held-out examples.
    pooled = spectra.reshape(len(spectra), 32, 4, 64, 4).mean(axis=(2,4))
    audio_mean, audio_std = float(pooled[tr].mean()), float(pooled[tr].std())
    audio = ((pooled - audio_mean) / max(audio_std, 1e-6)).astype("float32")
    lyrics = meta.lyrics.str.replace("\\n", " ", regex=False).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, min_df=2, max_df=.95,
                                 max_features=10000, ngram_range=(1,2), sublinear_tf=True)
    vectorizer.fit([lyrics[i] for i in tr])
    tfidf = vectorizer.transform(lyrics)
    svd = TruncatedSVD(n_components=text_dim, random_state=2026)
    svd.fit(tfidf[tr])
    text_scaler = StandardScaler().fit(svd.transform(tfidf[tr]))
    text = text_scaler.transform(svd.transform(tfidf)).astype("float32")
    transforms = {"audio_mean":audio_mean, "audio_std":audio_std,
                  "vectorizer":vectorizer, "svd":svd, "text_scaler":text_scaler}
    return audio[:,None], text, transforms
