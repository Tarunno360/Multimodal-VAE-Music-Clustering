"""Run the registered configuration; every numeric result is written from computation."""
import argparse
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_data, make_splits, prepare_features, sha256
from .models import MusicVAE, objective
from .metrics import evaluate, latent_diagnostics


def write_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def validation(model, a, t, c, beta, free_bits):
    model.eval()
    rng=torch.Generator().manual_seed(9173)
    total=[]
    with torch.no_grad():
        for _ in range(4 if not model.deterministic else 1):
            eps=torch.randn((len(a),model.latent),generator=rng)
            out=model(a,t,c,epsilon=eps)
            vals=objective(out,a,t,beta,free_bits,model.deterministic)
            total.append([float(v) for v in vals])
    return np.mean(total,axis=0).tolist()


def train_model(config, spec, seed, audio, text, conditions, splits, directory):
    seed_all(seed)
    directory.mkdir(parents=True,exist_ok=True)
    kwargs={"kind":spec["kind"],"latent":config["latent_dim"],"text_dim":text.shape[1],
            "condition_dim":conditions.shape[1] if spec.get("conditional") else 0,
            "deterministic":spec.get("deterministic",False)}
    model=MusicVAE(**kwargs)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["learning_rate"])
    a=torch.from_numpy(audio)
    t=torch.from_numpy(text)
    c=torch.from_numpy(conditions) if spec.get("conditional") else None
    tr,va=splits["train"],splits["val"]
    best=float("inf")
    best_state=None
    stale=0
    history=[]
    start=time.perf_counter()
    for epoch in range(1,config["epochs"]+1):
        model.train()
        beta=spec["beta"]*min(1.,epoch/max(config["warmup"],1))
        rows=[]
        for ix in torch.from_numpy(tr)[torch.randperm(len(tr))].split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            outputs=model(a[ix],t[ix],c[ix] if c is not None else None)
            vals=objective(outputs,a[ix],t[ix],beta,spec.get("free_bits",0),model.deterministic)
            if not torch.isfinite(vals[0]):
                raise FloatingPointError(f"Nonfinite loss at {spec['name']}, seed {seed}, epoch {epoch}")
            vals[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            rows.append([len(ix)]+[float(v.detach()) for v in vals])
        train=np.average(np.array(rows)[:,1:],axis=0,weights=np.array(rows)[:,0])
        val=validation(model,a[va],t[va],c[va] if c is not None else None,spec["beta"],spec.get("free_bits",0))
        record={"epoch":epoch,"beta":beta}
        for prefix,values in [("train",train),("val",val)]:
            record.update({prefix+"_"+k:float(v) for k,v in zip(["objective","audio_mse","text_mse","kl"],values)})
        history.append(record)
        # Wait for the target KL weight before checkpoint selection and early stopping.
        if epoch>=min(config["warmup"],config["epochs"]):
            if val[0]<best-1e-6:
                best=val[0]
                best_state=copy.deepcopy(model.state_dict())
                best_epoch=epoch
                stale=0
            else:
                stale+=1
        if epoch%20==0:
            print(f"  {spec['name']} seed={seed} epoch={epoch} val={val[0]:.5f}",flush=True)
        if stale>=config["patience"]:
            break
    model.load_state_dict(best_state)
    model.eval()
    pd.DataFrame(history).to_csv(directory/"history.csv",index=False)
    torch.save({"state_dict":best_state,"model_kwargs":kwargs,"spec":spec},directory/"model.pt")
    mus,lvs,arecs,trecs=[],[],[],[]
    with torch.no_grad():
        for ix in torch.arange(len(a)).split(128):
            mu,lv=model.encode(a[ix],t[ix],c[ix] if c is not None else None)
            ar,tx=model.decode(mu,c[ix] if c is not None else None)
            mus.append(mu.numpy());lvs.append(lv.numpy());arecs.append(ar.numpy())
            if tx is not None:
                trecs.append(tx.numpy())
    mu,lv,arec=np.concatenate(mus),np.concatenate(lvs),np.concatenate(arecs)
    summary={"best_epoch":best_epoch,"epochs_run":len(history),"best_val_objective":best,
             "parameters":sum(p.numel() for p in model.parameters()),"seconds":time.perf_counter()-start,
             "diagnostics":{}}
    for split,ix in splits.items():
        summary["diagnostics"][split]=latent_diagnostics(mu[ix],None if model.deterministic else lv[ix])
        summary["diagnostics"][split]["audio_mse_at_mean"]=float(np.mean((audio[ix]-arec[ix])**2))
        if trecs:
            summary["diagnostics"][split]["text_mse_at_mean"]=float(np.mean((text[ix]-np.concatenate(trecs)[ix])**2))
    np.savez_compressed(directory/"representations.npz",mu=mu,logvar=lv,audio_reconstruction=arec)
    write_json(directory/"training.json",summary)
    print(f"Finished {spec['name']} seed={seed}: epoch {best_epoch}, {summary['seconds']:.1f}s",flush=True)
    return mu, summary


def baselines(audio,text,splits,latent):
    tr=splits["train"]
    flat=audio.reshape(len(audio),-1)
    pa=PCA(n_components=latent,svd_solver="randomized",random_state=2026).fit(flat[tr])
    pt=PCA(n_components=latent,svd_solver="full").fit(text[tr])
    joined=np.concatenate([flat/np.sqrt(flat[tr].var(0).sum()),text/np.sqrt(text[tr].var(0).sum())],1)
    pf=PCA(n_components=latent,svd_solver="randomized",random_state=2026).fit(joined[tr])
    # Time mean/std of each of the 32 pooled frequency rows; no claim of MFCC extraction.
    stats=np.concatenate([audio[:,0].mean(2),audio[:,0].std(2)],1)
    ss=StandardScaler().fit(stats[tr]); stats=ss.transform(stats)
    ps=PCA(n_components=latent,svd_solver="full").fit(stats[tr])
    return {"audio_pca":pa.transform(flat),"lyrics_lsa":pt.transform(text),
            "fusion_pca":pf.transform(joined),"audio_statistics":ps.transform(stats)},\
           {"audio_pca":pa,"lyrics_pca":pt,"fusion_pca":pf,"stats_scaler":ss,"stats_pca":ps,
            "fusion_audio_norm":float(np.sqrt(flat[tr].var(0).sum())),
            "fusion_text_norm":float(np.sqrt(text[tr].var(0).sum()))}


def evaluate_representation(z, labels, splits, name, seed, config, directory, oracle=False):
    rows=[]
    tr,va,te=splits["train"],splits["val"],splits["test"]
    # A single scalar (not coordinate whitening) prevents arbitrary near-zero dimensions
    # being magnified. K-means and silhouette are invariant to this global rescaling.
    mean=z[tr].mean(0)
    scale=max(float(np.sqrt(z[tr].var(0).sum())),1e-8)
    z=(z-mean)/scale
    directory.mkdir(parents=True,exist_ok=True)
    for k in [2,3,4,5,6,8,10]:
        km=KMeans(n_clusters=k,n_init=20,random_state=seed).fit(z[tr])
        for split in ["val","test"]:
            ix=splits[split]
            pred=km.predict(z[ix])
            row={"method":name,"seed":seed,"split":split,"protocol":"inductive_kmeans",
                 "k_requested":k,"oracle":oracle,**evaluate(z[ix],pred,labels[ix])}
            rows.append(row)
            if k==config["clusters"]:
                np.save(directory/f"{split}_clusters.npy",pred)
        if k==config["clusters"]:
            joblib.dump({"kmeans":km,"mean":mean,"scale":scale},directory/"kmeans.joblib")
    # Ward and DBSCAN have no native predict: clearly separate within-test descriptions.
    pred=AgglomerativeClustering(n_clusters=config["clusters"],linkage="ward").fit_predict(z[te])
    rows.append({"method":name,"seed":seed,"split":"test","protocol":"transductive_ward",
                 "k_requested":config["clusters"],"oracle":oracle,**evaluate(z[te],pred,labels[te])})
    # Radius grid derives from TRAIN neighbors; select with label-free validation criterion.
    dist=NearestNeighbors(n_neighbors=5).fit(z[tr]).kneighbors(z[tr])[0][:,-1]
    grid=sorted(set(float(np.quantile(dist,q))*m for q in [.5,.9] for m in [.5,1.,2.]))
    candidates=[]
    for eps in grid:
        eps=max(eps,1e-8)
        pred=DBSCAN(eps=eps,min_samples=5).fit_predict(z[va])
        # Dummy constant labels: selection cannot access genre information.
        met=evaluate(z[va],pred,np.zeros(len(va)))
        score=-2. if met["silhouette"] is None else met["coverage"]*met["silhouette"]-(1-met["coverage"])
        candidates.append({"eps":eps,"score":score,"coverage":met["coverage"],"n_clusters":met["n_clusters"]})
    chosen=max(candidates,key=lambda x:x["score"])
    pred=DBSCAN(eps=chosen["eps"],min_samples=5).fit_predict(z[te])
    rows.append({"method":name,"seed":seed,"split":"test","protocol":"transductive_dbscan",
                 "k_requested":None,"oracle":oracle,"eps":chosen["eps"],**evaluate(z[te],pred,labels[te])})
    write_json(directory/"dbscan_selection.json",candidates)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default=".")
    parser.add_argument("--config",default="configs/main.json")
    parser.add_argument("--output",default="results/rebuilt_v1")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    root=Path(args.root).resolve(); out=(root/args.output).resolve()
    config=json.loads((root/args.config).read_text())
    torch.set_num_threads(config["threads"])
    config_sources={str(p.relative_to(root)):sha256(p) for p in sorted((root/"musicvae").glob("*.py"))}
    spectra,meta,groups,manifest=load_data(root)
    signature=hashlib.sha256(json.dumps({"config":config,"data":manifest,"code":config_sources},sort_keys=True).encode()).hexdigest()
    if (out/"manifest.json").exists():
        prior=json.loads((out/"manifest.json").read_text())
        if not args.resume or prior["signature"]!=signature:
            raise ValueError("Output exists or run signature changed; use a new output directory (or --resume for identical runs)")
    out.mkdir(parents=True,exist_ok=True)
    splits=make_splits(groups,config["split_seed"])
    split_names=np.full(len(meta),"",dtype=object)
    for s,ix in splits.items():split_names[ix]=s
    pd.DataFrame({"row":np.arange(len(meta)),"id":meta.id,"group":groups,"split":split_names,"genre":meta.genre}).to_csv(out/"splits.csv",index=False)
    with threadpool_limits(limits=config["threads"]):
        audio,text,transforms=prepare_features(spectra,meta,splits,config["text_dim"])
        base,base_transforms=baselines(audio,text,splits,config["latent_dim"])
    le=LabelEncoder().fit(meta.genre.iloc[splits["train"]])
    labels=le.transform(meta.genre)
    conditions=np.eye(len(le.classes_),dtype="float32")[labels]
    joblib.dump({**transforms,**base_transforms,"genres":le.classes_},out/"preprocessing.joblib")
    np.savez_compressed(out/"features.npz",audio=audio,text=text,labels=labels,**base)
    write_json(out/"manifest.json",{"signature":signature,"config":config,"sources":config_sources,
        "data":manifest,"platform":platform.platform(),"python":platform.python_version(),"device":"cpu",
        "packages":{k:importlib.metadata.version(k) for k in ["torch","numpy","pandas","scikit-learn","scipy","matplotlib"]},
        "split_sizes":{k:len(v) for k,v in splits.items()},"genres":le.classes_.tolist(),
        "split_genres":{s:meta.iloc[ix].genre.value_counts().to_dict() for s,ix in splits.items()},
        "groups":int(len(np.unique(groups))),"text_vocabulary":len(transforms["vectorizer"].vocabulary_),
        "text_svd_train_explained_variance":float(transforms["svd"].explained_variance_ratio_.sum())})
    allrows=[]
    for seed in config["seeds"]:
        for name,z in base.items():
            with threadpool_limits(limits=config["threads"]):
                allrows+=evaluate_representation(z,labels,splits,name,seed,config,out/name/str(seed))
        for spec in config["models"]:
            directory=out/spec["name"]/str(seed)
            if args.resume and (directory/"training.json").exists():
                z=np.load(directory/"representations.npz")["mu"]
            else:
                model_text=text.copy()
                if spec.get("shuffle_text"):
                    # Split-local permutations avoid train/test contamination.
                    rng=np.random.default_rng(seed+9000)
                    for ix in splits.values():model_text[ix]=text[rng.permutation(ix)]
                z,summary=train_model(config,spec,seed,audio,model_text,conditions,splits,directory)
            with threadpool_limits(limits=config["threads"]):
                allrows+=evaluate_representation(z,labels,splits,spec["name"],seed,config,directory,spec.get("conditional",False))
            write_json(out/"metrics.json",allrows)
            pd.DataFrame(allrows).to_csv(out/"metrics.csv",index=False)
    write_json(out/"completion.json",{"signature":signature,"completed":True,"models":len(config["models"]),"seeds":config["seeds"]})
    print(f"Complete: {len(allrows)} evaluation records saved in {out}",flush=True)


if __name__=="__main__":
    main()
