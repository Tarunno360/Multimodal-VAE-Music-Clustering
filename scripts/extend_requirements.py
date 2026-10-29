"""Additional rubric checks requested after the initial experiment was frozen."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from musicvae.experiment import train_model, evaluate_representation, write_json
from musicvae.data import sha256


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/"results/rebuilt_v1";out=root/"results/requirements_extension"
    if (out/"completion.json").exists():
        print("Requirement extension is already complete.");return
    manifest=json.loads((source/"manifest.json").read_text());c=manifest["config"]
    torch.set_num_threads(c["threads"])
    data=np.load(source/"features.npz");a,t,y=data["audio"],data["text"],data["labels"]
    frame=pd.read_csv(source/"splits.csv")
    splits={s:np.flatnonzero(frame.split.to_numpy()==s) for s in ["train","val","test"]}
    cond=np.eye(len(manifest["genres"]),dtype="float32")[y]
    spec={"name":"multimodal_conditional_oracle","kind":"multimodal","beta":.01,"conditional":True}
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"manifest.json",{"parent_signature":manifest["signature"],"script_sha256":sha256(__file__),
         "reason":"Actual assignment supplied after primary protocol was frozen", "models":[spec],"config":c})
    rows=[]
    for seed in c["seeds"]:
        for name,z in {"direct_spectral":a.reshape(len(a),-1)}.items():
            with threadpool_limits(limits=c["threads"]):
                rows+=evaluate_representation(z,y,splits,name,seed,c,out/name/str(seed))
        directory=out/spec["name"]/str(seed)
        z,_=train_model(c,spec,seed,a,t,cond,splits,directory)
        with threadpool_limits(limits=c["threads"]):
            rows+=evaluate_representation(z,y,splits,spec["name"],seed,c,directory,True)
        write_json(out/"metrics.json",rows)
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    write_json(out/"completion.json",{"completed":True,"parent_signature":manifest["signature"]})


if __name__=="__main__":main()
