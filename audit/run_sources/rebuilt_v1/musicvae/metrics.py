"""Clustering metrics with explicit treatment of noise and degenerate partitions."""
import numpy as np
from sklearn.metrics import (silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
                             adjusted_mutual_info_score, davies_bouldin_score, calinski_harabasz_score)
from sklearn.metrics.cluster import contingency_matrix


def evaluate(features, clusters, labels):
    features=np.asarray(features)
    clusters=np.asarray(clusters)
    labels=np.asarray(labels)
    if len(features)!=len(clusters) or len(labels)!=len(clusters):
        raise ValueError("Features, predictions and labels must have identical row counts")
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite features")
    mask=clusters!=-1
    x,c,y=features[mask],clusters[mask],labels[mask]
    _,counts=np.unique(c,return_counts=True)
    n,k=len(c),len(counts)
    out={"n":len(clusters),"coverage":float(mask.mean()),"n_clusters":k,
         "cluster_sizes":counts.tolist(), "silhouette":None,"davies_bouldin":None,
         "calinski_harabasz":None,"ari":None,"nmi":None,"ami":None,"purity":None,
         "largest_fraction":None,"balance_ratio":None,"effective_clusters":None,
         "majority_baseline":None}
    if n:
        p=counts/n
        out.update(largest_fraction=float(p.max()),balance_ratio=float(counts.min()/counts.max()),
                   effective_clusters=float(np.exp(-np.sum(p*np.log(p)))),
                   ari=float(adjusted_rand_score(y,c)),nmi=float(normalized_mutual_info_score(y,c)),
                   ami=float(adjusted_mutual_info_score(y,c)),
                   purity=float(contingency_matrix(y,c).max(axis=0).sum()/n),
                   majority_baseline=float(np.unique(y,return_counts=True)[1].max()/n))
    if 1<k<n:
        out.update(silhouette=float(silhouette_score(x,c)),
                   davies_bouldin=float(davies_bouldin_score(x,c)),
                   calinski_harabasz=float(calinski_harabasz_score(x,c)))
    return out


def latent_diagnostics(mu, logvar=None):
    centered=mu-mu.mean(0)
    s=np.linalg.svd(centered,compute_uv=False)**2
    p=s/s.sum() if s.sum()>0 else np.zeros_like(s)
    p=p[p>0]
    out={"active_units":int((mu.var(0)>.01).sum()),
         "mean_latent_variance":float(mu.var(0).mean()),
         "effective_rank":float(np.exp(-(p*np.log(p)).sum())) if len(p) else 0.}
    if logvar is not None:
        out["kl_per_dimension"]=float(.5*np.mean(mu**2+np.exp(logvar)-1-logvar))
    return out
