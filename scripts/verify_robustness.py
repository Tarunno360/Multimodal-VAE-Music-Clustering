"""Reload every follow-up checkpoint and independently recompute saved k=5 metrics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits
from musicvae.models import MusicVAE
from musicvae.metrics import evaluate
from musicvae.data import sha256
from musicvae.experiment import write_json

def main():
 torch.set_num_threads(4);models=0;metrics=0;rows=0
 for split_seed in [2027,2028]:
  p=ROOT/f'results/robustness_{split_seed}'
  assert json.loads((p/'completion.json').read_text())['completed']
  m=json.loads((p/'manifest.json').read_text())
  for path,digest in m['sources'].items():assert sha256(ROOT/path)==digest,path
  for item in m['data'].values():assert sha256(ROOT/item['path'])==item['sha256']
  split=pd.read_csv(p/'splits.csv')
  splits={s:np.flatnonzero(split.split.to_numpy()==s) for s in ['train','val','test']}
  for a in splits:
   for b in splits:
    if a!=b:assert not set(split.iloc[splits[a]].group)&set(split.iloc[splits[b]].group)
  f=np.load(p/'features.npz');audio=f['audio'];text=f['text'];labels=f['labels']
  d=pd.read_csv(p/'metrics.csv');rows+=len(d)
  for checkpoint in sorted(p.glob('*/*/model.pt')):
   ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
   model=MusicVAE(**ck['model_kwargs']);model.load_state_dict(ck['state_dict']);model.eval()
   t=text.copy();seed=int(checkpoint.parent.name)
   if ck['spec'].get('shuffle_text'):
    rng=np.random.default_rng(seed+9000)
    for ix in splits.values():t[ix]=text[rng.permutation(ix)]
   encoded=[]
   with torch.no_grad():
    for ix in np.array_split(np.arange(len(audio)),np.arange(128,len(audio),128)):
     encoded.append(model.encode(torch.from_numpy(audio[ix]),torch.from_numpy(t[ix]))[0].numpy())
   z=np.concatenate(encoded)
   np.testing.assert_allclose(z,np.load(checkpoint.parent/'representations.npz')['mu'],rtol=1e-5,atol=1e-5)
   fitted=joblib.load(checkpoint.parent/'kmeans.joblib');z=(z-fitted['mean'])/fitted['scale']
   for s,ix in splits.items():
    if s!='train':np.testing.assert_array_equal(fitted['kmeans'].predict(z[ix]),np.load(checkpoint.parent/f'{s}_clusters.npy'))
   models+=1
  for fit in sorted(p.glob('*/*/kmeans.joblib')):
   method=fit.parent.parent.name;seed=int(fit.parent.name)
   z=np.load(fit.parent/'representations.npz')['mu'] if (fit.parent/'representations.npz').exists() else f[method]
   fitted=joblib.load(fit);z=(z-fitted['mean'])/fitted['scale'];ix=splits['test']
   pred=np.load(fit.parent/'test_clusters.npy')
   with threadpool_limits(limits=4):actual=evaluate(z[ix],pred,labels[ix])
   row=d[(d.method==method)&(d.seed==seed)&(d.protocol=='inductive_kmeans')&(d.split=='test')&(d.k_requested==5)].iloc[0]
   for metric in ['ari','nmi','silhouette','purity','davies_bouldin','calinski_harabasz','largest_fraction']:
    assert np.isclose(actual[metric],row[metric],rtol=1e-5,atol=1e-6),(split_seed,method,seed,metric)
   metrics+=1
 assert (models,metrics,rows)==(60,100,1600),(models,metrics,rows)
 report={'passed':True,'neural_checkpoints_reloaded':models,'heldout_metrics_recomputed':metrics,
         'metric_rows':rows,'split_seeds':[2027,2028],'source_and_data_hashes_verified':True,'split_groups_disjoint':True}
 write_json(ROOT/'audit/robustness_verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
