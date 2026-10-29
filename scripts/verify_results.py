"""Verify delivered experiment integrity and checkpoint inference without retraining."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits
from musicvae.models import MusicVAE
from musicvae.metrics import evaluate
from musicvae.data import sha256
from musicvae.experiment import write_json

def main():
 torch.set_num_threads(4)
 p=ROOT/'results/rebuilt_v1';ext=ROOT/'results/requirements_extension'
 m=json.loads((p/'manifest.json').read_text())
 for relative,digest in m['sources'].items():assert sha256(ROOT/relative)==digest,relative
 for item in m['data'].values():assert sha256(ROOT/item['path'])==item['sha256'],item['path']
 split=pd.read_csv(p/'splits.csv');splits={s:np.flatnonzero(split.split.to_numpy()==s) for s in ['train','val','test']}
 for a in splits:
  for b in splits:
   if a!=b:assert not set(split.iloc[splits[a]].group)&set(split.iloc[splits[b]].group)
 f=np.load(p/'features.npz');audio=f['audio'];text=f['text'];labels=f['labels'];conds=np.eye(5,dtype='float32')[labels]
 frames=[pd.read_csv(q/'metrics.csv') for q in [p,ext]]
 d=pd.concat(frames,ignore_index=True);count=0;checked_models=0
 for source in [p,ext]:
  assert json.loads((source/'completion.json').read_text())['completed']
  for checkpoint in sorted(source.glob('*/*/model.pt')):
   ck=torch.load(checkpoint,map_location='cpu',weights_only=True);model=MusicVAE(**ck['model_kwargs']);model.load_state_dict(ck['state_dict']);model.eval()
   spec=ck['spec'];seed=int(checkpoint.parent.name);t=text.copy()
   if spec.get('shuffle_text'):
    rng=np.random.default_rng(seed+9000)
    for ix in splits.values():t[ix]=text[rng.permutation(ix)]
   encoded=[]
   with torch.no_grad():
    for ix in np.array_split(np.arange(len(audio)),np.arange(128,len(audio),128)):
     c=torch.from_numpy(conds[ix]) if ck['model_kwargs']['condition_dim'] else None
     mu,_=model.encode(torch.from_numpy(audio[ix]),torch.from_numpy(t[ix]),c);encoded.append(mu.numpy())
   z=np.concatenate(encoded);saved=np.load(checkpoint.parent/'representations.npz')['mu']
   np.testing.assert_allclose(z,saved,rtol=1e-5,atol=1e-5)
   fitted=joblib.load(checkpoint.parent/'kmeans.joblib');zt=(z-fitted['mean'])/fitted['scale']
   for s in ['val','test']:
    ix=splits[s];prediction=fitted['kmeans'].predict(zt[ix]);reference=np.load(checkpoint.parent/f'{s}_clusters.npy');np.testing.assert_array_equal(prediction,reference)
   checked_models+=1
  for fit in sorted(source.glob('*/*/kmeans.joblib')):
   name=fit.parent.parent.name;seed=int(fit.parent.name)
   if (fit.parent/'representations.npz').exists():z=np.load(fit.parent/'representations.npz')['mu']
   elif name=='direct_spectral':z=audio.reshape(len(audio),-1)
   else:z=f[name]
   fitted=joblib.load(fit);z=(z-fitted['mean'])/fitted['scale'];ix=splits['test'];pred=np.load(fit.parent/'test_clusters.npy')
   with threadpool_limits(limits=4):actual=evaluate(z[ix],pred,labels[ix])
   row=d[(d.method==name)&(d.seed==seed)&(d.protocol=='inductive_kmeans')&(d.split=='test')&(d.k_requested==5)].iloc[0]
   for metric in ['ari','nmi','silhouette','purity','davies_bouldin','calinski_harabasz','largest_fraction']:
    assert np.isclose(actual[metric],row[metric],rtol=1e-5,atol=1e-6),(name,seed,metric)
   count+=1
 assert checked_models==60 and count==85,(checked_models,count)
 bm=json.loads((ROOT/'data/bilingual/manifest.json').read_text());assert len(bm['rows'])==40
 for row in bm['rows']:assert sha256(ROOT/row['audio_path'])==row['sha256'],row['id']
 assert json.loads((ROOT/'results/bilingual/completion.json').read_text())['completed']
 # Independently reload the five bilingual models as well.
 bdir=ROOT/'results/bilingual'
 bmeta=json.loads((bdir/'manifest.json').read_text())
 raw=np.load(ROOT/'data/bilingual/prepared/spectrograms.npy')
 ba=raw.reshape(40,32,4,64,4).mean((2,4))
 ba=((ba-bmeta['audio_mean'])/bmeta['audio_std']).astype('float32')[:,None]
 by=np.array([0 if r['language']=='en' else 1 for r in bm['rows']])
 bmetrics=pd.read_csv(bdir/'metrics.csv')
 from sklearn.cluster import KMeans
 for seed in [11,22,33,44,55]:
  bd=bdir/'audio_vae'/str(seed);ck=torch.load(bd/'model.pt',map_location='cpu',weights_only=True)
  model=MusicVAE(**ck['model_kwargs']);model.load_state_dict(ck['state_dict']);model.eval()
  with torch.no_grad():bz=model.encode(torch.from_numpy(ba))[0].numpy()
  np.testing.assert_allclose(bz,np.load(bd/'representations.npz')['mu'],rtol=1e-5,atol=1e-5)
  for k in [2,5]:
   with threadpool_limits(limits=4):
    bc=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(bz)
    actual=evaluate(bz,bc,by)
   recorded=bmetrics[(bmetrics.method=='VAE')&(bmetrics.seed==seed)&(bmetrics.k==k)].iloc[0]
   for metric in ['ari','nmi','silhouette','calinski_harabasz']:
    assert np.isclose(actual[metric],recorded[metric],rtol=1e-5,atol=1e-6)
 report={'passed':True,'input_hashes_verified':len(m['data']),'core_source_hashes_verified':len(m['sources']),
 'main_and_extension_checkpoints_reloaded':checked_models,'heldout_clustering_results_recomputed':count,
 'bilingual_checkpoints_reloaded':5,'bilingual_vae_metrics_recomputed':10,
 'bilingual_recording_hashes_verified':40,'split_groups_disjoint':True,'raw_metric_rows':len(d)+len(pd.read_csv(ROOT/'results/bilingual/metrics.csv'))}
 write_json(ROOT/'audit/verification.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
