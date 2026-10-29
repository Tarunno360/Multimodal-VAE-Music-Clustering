"""Generate manuscript tables and plots from completed experiment files only."""
from pathlib import Path
import sys,os,json,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score,silhouette_score
from threadpoolctl import threadpool_limits
from musicvae.models import MusicVAE
from musicvae.data import sha256
from musicvae.experiment import write_json
P=ROOT/'results/rebuilt_v1';E=ROOT/'results/requirements_extension';OUT=ROOT/'results/analysis'
TEX=ROOT/'Final Writing Latex/generated';FIG=ROOT/'Final Writing Latex/figures'
LABELS={'audio_pca':'Audio PCA','lyrics_lsa':'Lyrics LSA','fusion_pca':'Fusion PCA','audio_statistics':'Audio statistics',
'direct_spectral':'Direct spectral','audio_ae':'Audio AE','audio_vae_b001':'Audio VAE (0.001)','audio_vae_b01':'Audio VAE (0.01)',
'audio_vae_b1':'Audio VAE (0.1)','audio_vae_b100':'Audio VAE (1)','conv_vae':'Conv VAE','multimodal_vae':'Multimodal VAE',
'hybrid_vae':'Hybrid VAE','multimodal_freebits':'Multimodal + FB','multimodal_shuffled':'Shuffled text',
'conditional_oracle':'Audio oracle','multimodal_conditional_oracle':'Multimodal oracle'}
ORDER=list(LABELS)
PRIMARY=['direct_spectral','audio_pca','audio_statistics','lyrics_lsa','fusion_pca','audio_ae','audio_vae_b01','conv_vae','multimodal_vae','hybrid_vae']
SEEDS=[11,22,33,44,55]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':160})

def source(name):return E if name in ['direct_spectral','multimodal_conditional_oracle'] else P

def savefig(name):
 plt.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
 plt.savefig(FIG/(name+'.png'),bbox_inches='tight',dpi=180)
 plt.close()

def meanstd(frame,method,metric):
 a=frame[frame.method==method][metric].dropna().to_numpy()
 return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else 0.

def fmt(frame,name,metric):
 a,b=meanstd(frame,name,metric)
 return f'${a:.3f}\\pm{b:.3f}$'

def table(filename,caption,label,header,rows):
 s='\\begin{table}[htbp]\n\\centering\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\small\n'
 s+='\\begin{tabular}{l'+'r'*(len(header)-1)+'}\n\\toprule\n'+' & '.join(header)+' \\\\\n\\midrule\n'
 s+='\n'.join(' & '.join(map(str,r))+' \\\\' for r in rows)
 s+='\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
 (TEX/filename).write_text(s)

def main():
 for folder in [OUT,TEX,FIG]:folder.mkdir(parents=True,exist_ok=True)
 for folder in [P,E]:
  if not (folder/'completion.json').exists():raise RuntimeError(f'Incomplete experiment {folder}')
 manifest=json.loads((P/'manifest.json').read_text())
 d=pd.concat([pd.read_csv(P/'metrics.csv'),pd.read_csv(E/'metrics.csv')],ignore_index=True)
 test=d[(d.split=='test')&(d.protocol=='inductive_kmeans')&(d.k_requested==5)]
 assert len(test)==len(ORDER)*5
 test.groupby('method').agg({k:['mean','std'] for k in ['ari','nmi','silhouette','davies_bouldin','calinski_harabasz','purity','largest_fraction','effective_clusters']}).to_csv(OUT/'summary.csv')
 splits=pd.read_csv(P/'splits.csv');te=np.flatnonzero(splits.split.to_numpy()=='test')
 features=np.load(P/'features.npz'); y=features['labels'][te];genres=manifest['genres']
 pred={name:np.stack([np.load(source(name)/name/str(seed)/'test_clusters.npy') for seed in SEEDS]) for name in ORDER}
 diagnostics=[]
 for name in ORDER:
  for seed in SEEDS:
   f=source(name)/name/str(seed)/'training.json'
   if f.exists():
    x=json.loads(f.read_text());diagnostics.append({'method':name,'seed':seed,**x['diagnostics']['train'],
     'test_audio_mse':x['diagnostics']['test']['audio_mse_at_mean'],'epoch':x['best_epoch'],'seconds':x['seconds'],'parameters':x['parameters']})
 diag=pd.DataFrame(diagnostics);diag.to_csv(OUT/'training_diagnostics.csv',index=False)
 # Fixed models, paired test-group and optimization-seed bootstrap. No model is refitted.
 pairs=[('hybrid_vae','audio_vae_b01'),('hybrid_vae','lyrics_lsa'),('hybrid_vae','fusion_pca'),
        ('multimodal_vae','multimodal_shuffled'),('multimodal_freebits','multimodal_vae'),('audio_vae_b01','audio_pca')]
 rng=np.random.default_rng(20261029)
 g=splits.iloc[te].group.to_numpy();units=[np.flatnonzero(g==u) for u in np.unique(g)]
 boots={pair:[] for pair in pairs}
 for b in range(1000):
  ix=np.concatenate([units[u] for u in rng.integers(len(units),size=len(units))]);se=rng.integers(5,size=5)
  ari={name:np.mean([adjusted_rand_score(y[ix],pred[name][s,ix]) for s in se]) for name in set(itertools.chain.from_iterable(pairs))}
  for pair in pairs:boots[pair].append(ari[pair[0]]-ari[pair[1]])
 boot=[]
 for a,b in pairs:
  lo,hi=np.quantile(boots[(a,b)],[.025,.975])
  boot.append({'a':a,'b':b,'mean_difference':meanstd(test,a,'ari')[0]-meanstd(test,b,'ari')[0],'lower95':float(lo),'upper95':float(hi),'resamples':1000})
 write_json(OUT/'paired_bootstrap.json',boot)
 stability=[]
 common=features['audio_pca'][te]
 for name in ORDER:
  scores=[adjusted_rand_score(pred[name][i],pred[name][j]) for i,j in itertools.combinations(range(5),2)]
  stability.append({'method':name,'mean_pairwise_seed_ari':np.mean(scores),'common_audio_space_silhouette':np.mean([silhouette_score(common,p) for p in pred[name]])})
 pd.DataFrame(stability).to_csv(OUT/'stability_and_common_space.csv',index=False)
 # Table exports: all values originate from metric files.
 table('main_table.tex','Inductive K-means on held-out songs, $k=5$. Mean $\\pm$ sample SD over five seeds; higher is better for the displayed metrics.','tab:main',
       ['Representation','ARI','NMI','Silhouette','Purity'],[[LABELS[n]]+[fmt(test,n,m) for m in ['ari','nmi','silhouette','purity']] for n in PRIMARY])
 ab=['audio_vae_b001','audio_vae_b01','audio_vae_b1','audio_vae_b100','multimodal_vae','multimodal_freebits','multimodal_shuffled']
 table('ablation_table.tex','Regularization and pairing controls on the same held-out split. FB denotes a 0.05-nat free-bits threshold per latent dimension.','tab:ablation',
       ['Representation','ARI','Silhouette','Largest share'],[[LABELS[n]]+[fmt(test,n,m) for m in ['ari','silhouette','largest_fraction']] for n in ab])
 table('full_table.tex','Additional primary metrics, mean $\\pm$ SD; DB is lower-better, CH and purity are higher-better. Label-conditioned oracles are shown separately.','tab:full',
       ['Representation','DB','CH','Purity'],[[LABELS[n]]+[fmt(test,n,m) for m in ['davies_bouldin','calinski_harabasz','purity']] for n in ORDER if 'oracle' not in n])
 table('diagnostics_table.tex','Training latent diagnostics and held-out reconstruction MSE at the posterior mean. AU counts variance above 0.01.','tab:diagnostics',
       ['Representation','AU','KL/dim','Test MSE'],[[LABELS[n]]+[fmt(diag,n,m) for m in ['active_units','kl_per_dimension','test_audio_mse']] for n in ORDER if n in diag.method.unique() and n!='audio_ae' and 'oracle' not in n])
 td=d[(d.split=='test')&(d.protocol!='inductive_kmeans')]
 tdrows=[]
 for n in ['audio_pca','lyrics_lsa','conv_vae','hybrid_vae','multimodal_vae']:
  for proto,plabel in [('transductive_ward','Ward'),('transductive_dbscan','DBSCAN')]:
   q=td[td.protocol==proto];a=q[q.method==n]
   tdrows.append([LABELS[n]+' '+plabel,fmt(q,n,'coverage'),fmt(q,n,'n_clusters'),fmt(q,n,'ari') if a.ari.notna().any() else '--',fmt(q,n,'silhouette') if a.silhouette.notna().any() else '--'])
 table('transductive_table.tex','Within-test clustering (not an inductive comparison). Coverage is the fraction not labeled noise. Undefined scores are shown as --; when only some runs are valid, metrics average valid runs.','tab:transductive',
       ['Representation / method','Coverage','Clusters','ARI','Silhouette'],tdrows)
 table('bootstrap_table.tex','Exploratory paired ARI differences: 1,000 resamples of test duplicate groups and seeds; percentile intervals conditional on this one split. No multiple-comparison adjustment or model refitting.','tab:bootstrap',
       ['Comparison (A minus B)','Difference','95\\% interval'],[[LABELS[b['a']]+' - '+LABELS[b['b']],f"{b['mean_difference']:.3f}",f"[{b['lower95']:.3f}, {b['upper95']:.3f}]"] for b in boot])
 table('oracle_table.tex','Genre-conditioned diagnostic models. Both receive true genre at inference and cannot be interpreted as unsupervised methods.','tab:oracle',
       ['Oracle','ARI','NMI','Silhouette'],[[LABELS[n]]+[fmt(test,n,m) for m in ['ari','nmi','silhouette']] for n in ['conditional_oracle','multimodal_conditional_oracle']])
 old=pd.read_csv(ROOT/'audit/legacy_recomputed.csv')
 names={'1/results/latent_features_vae.npy':'Dense VAE','1/results/latent_features_conv.npy':'Conv VAE','2/results/latent_features_beta.npy':'Beta-VAE','3/results/latent_features_cvae.npy':'Conditional VAE','3/results/latent_features_pca.npy':'PCA'}
 table('legacy_table.tex','Recomputed legacy K-means results on all 683 saved embeddings, seed 42. These are in-sample diagnostics, not comparable held-out scores.','tab:legacy',
       ['Saved representation','Silhouette','ARI','Largest share'],[[names[r.file]]+[f'{getattr(r,m):.3f}' for m in ['silhouette','ari','largest_fraction']] for r in old.itertuples() if r.file in names])
 (TEX/'numbers.tex').write_text(''.join('\\newcommand{\\N'+key.capitalize()+'}{'+str(manifest['split_sizes'][key])+'}\n' for key in ['train','val','test']))
 h,hs=meanstd(test,'hybrid_vae','ari');l,ls=meanstd(test,'lyrics_lsa','ari');a,ast=meanstd(test,'audio_vae_b01','ari');pc,pcs=meanstd(test,'audio_pca','ari')
 (TEX/'abstract_results.tex').write_text(f'The convolutional audio--lyrics hybrid reaches ARI ${h:.3f}\\pm{hs:.3f}$, compared with ${a:.3f}\\pm{ast:.3f}$ for an audio VAE and ${l:.3f}\\pm{ls:.3f}$ for text-only features. The results support a benefit over audio alone, but do not establish an advantage over text alone.\n')
 b=boot[0];b2=boot[1]
 (TEX/'results_narrative.tex').write_text(f'Table~\\ref{{tab:main}} shows a hybrid ARI of ${h:.3f}\\pm{hs:.3f}$ against ${a:.3f}\\pm{ast:.3f}$ for the audio VAE at $\\beta=0.01$ and ${pc:.3f}\\pm{pcs:.3f}$ for audio PCA. The paired hybrid-minus-audio-VAE difference is {b["mean_difference"]:.3f}, with an exploratory 95\\% bootstrap interval [{b["lower95"]:.3f}, {b["upper95"]:.3f}]. Text-only LSA obtains ${l:.3f}\\pm{ls:.3f}$ ARI. The hybrid-minus-text difference is {b2["mean_difference"]:.3f}, interval [{b2["lower95"]:.3f}, {b2["upper95"]:.3f}], so these data do not establish a hybrid advantage over text. Test-set majority-class purity is {np.bincount(y).max()/len(y):.3f}. The comparison concerns a fixed corpus and split, not general superiority of one architecture.\n')
 mm,mms=meanstd(test,'multimodal_vae','ari');sh,shs=meanstd(test,'multimodal_shuffled','ari');fb,fbs=meanstd(test,'multimodal_freebits','ari');hi,_=meanstd(test,'audio_vae_b100','silhouette');hiari,_=meanstd(test,'audio_vae_b100','ari')
 (TEX/'ablation_narrative.tex').write_text(f'The dense multimodal model obtains ARI ${mm:.3f}\\pm{mms:.3f}$, whereas shuffled pairing obtains ${sh:.3f}\\pm{shs:.3f}$. The negative control indicates that preserving the relation between a song and its lyrics matters in this setup. Free bits gives ${fb:.3f}\\pm{fbs:.3f}$, but the large run variation prevents a robust superiority claim. Raising audio $\\beta$ to 1 increases mean silhouette to {hi:.3f} while ARI remains {hiari:.3f}. Figure~\\ref{{fig:beta}} reports the associated latent and reconstruction changes; a higher geometry score alone is not evidence of improvement.\n')
 (TEX/'compute_sentence.tex').write_text(f'The {len(diag)} trained neural-model runs in the primary study and requirement extension took {diag.seconds.sum()/60:.1f} minutes of summed local training wall time, excluding environment setup, clustering, reporting and the separate bilingual study. Selected epochs ranged from {int(diag.epoch.min())} to {int(diag.epoch.max())}; parameter counts ranged from {int(diag.parameters.min()):,} to {int(diag.parameters.max()):,}.\n')
 # Performance plot.
 fig,axes=plt.subplots(1,2,figsize=(8.2,3.9),sharey=True,layout='constrained')
 for ax,metric,title in zip(axes,['ari','silhouette'],['Genre agreement (ARI)','Silhouette in each feature space']):
  vals=[meanstd(test,n,metric) for n in PRIMARY]
  ax.barh(np.arange(len(PRIMARY)),[v[0] for v in vals],xerr=[v[1] for v in vals],color=['#157f79' if n in ['hybrid_vae','lyrics_lsa'] else '#617d9b' for n in PRIMARY],capsize=2)
  ax.set_yticks(np.arange(len(PRIMARY)),[LABELS[n] for n in PRIMARY]);ax.set_title(title);ax.grid(axis='x',alpha=.2)
 axes[0].invert_yaxis();savefig('rebuilt_performance')
 fig,axes=plt.subplots(1,4,figsize=(8.2,2.35),layout='constrained')
 bn=['audio_vae_b001','audio_vae_b01','audio_vae_b1','audio_vae_b100'];betas=[.001,.01,.1,1]
 for ax,frame,metric,title in zip(axes,[test,diag,diag,diag],['ari','test_audio_mse','kl_per_dimension','active_units'],['Test ARI','Test audio MSE','Train KL / dimension','Train active units']):
  vs=[meanstd(frame,n,metric) for n in bn];ax.errorbar(betas,[v[0] for v in vs],yerr=[v[1] for v in vs],marker='o',color='#157f79',capsize=2);ax.set_xscale('log');ax.set_title(title);ax.set_xlabel('beta');ax.grid(alpha=.2)
 savefig('rebuilt_beta')
 # Reconstructions selected by explicit rule, no cherry-picked best cases.
 sample=[te[np.flatnonzero(y==i)[0]] for i in range(len(genres))]
 recon=[('Input',features['audio'])]+[(LABELS[n],np.load(P/n/'11/representations.npz')['audio_reconstruction']) for n in ['audio_ae','audio_vae_b01','conv_vae','hybrid_vae']]
 fig,axes=plt.subplots(len(recon),len(sample),figsize=(8.2,4.8),layout='constrained')
 for i,(name,arr) in enumerate(recon):
  for j,ix in enumerate(sample):
   axes[i,j].imshow(arr[ix,0],aspect='auto',origin='lower',vmin=-3,vmax=3,cmap='magma');axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
   if i==0:axes[i,j].set_title(genres[j])
   if j==0:axes[i,j].set_ylabel(name.replace('Audio VAE (0.01)','Audio VAE'),rotation=0,ha='right',va='center',fontsize=10)
 savefig('rebuilt_reconstruction')
 fig,axes=plt.subplots(2,2,figsize=(7.8,5.4),layout='constrained')
 for ax,n in zip(axes.flat,['audio_pca','lyrics_lsa','audio_vae_b01','hybrid_vae']):
  z=features[n] if n in features.files else np.load(P/n/'11/representations.npz')['mu']
  with threadpool_limits(limits=4):xy=TSNE(n_components=2,perplexity=20,random_state=11,init='pca',learning_rate='auto').fit_transform(z[te])
  for i,genre in enumerate(genres):ax.scatter(xy[y==i,0],xy[y==i,1],s=15,label=genre,alpha=.8)
  ax.set_title(LABELS[n]);ax.set_xticks([]);ax.set_yticks([])
 axes[0,0].legend(fontsize=9,ncol=2);savefig('rebuilt_tsne')
 fig,axes=plt.subplots(1,2,figsize=(8.0,2.8),layout='constrained')
 for ax,n in zip(axes,['hybrid_vae','lyrics_lsa']):
  mat=np.zeros((5,len(genres)),int)
  for cl,genre in zip(pred[n][0],y):mat[cl,genre]+=1
  ax.imshow(mat,cmap='Blues',aspect='auto')
  for (i,j),v in np.ndenumerate(mat):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>mat.max()/2 else 'black',fontsize=11)
  ax.set_xticks(range(len(genres)),genres);ax.set_yticks(range(5));ax.set_ylabel('Cluster');ax.set_title(LABELS[n])
 savefig('rebuilt_genre_distribution')
 fig,ax=plt.subplots(figsize=(8.2,5.8),layout='constrained');left=np.zeros(len(ORDER))
 for c in range(5):
  vals=[int((pred[n][0]==c).sum()) for n in ORDER];ax.barh(range(len(ORDER)),vals,left=left,label=str(c));left+=vals
 ax.set_yticks(range(len(ORDER)),[LABELS[n]+(' (oracle)' if 'oracle' in n else '') for n in ORDER]);ax.invert_yaxis();ax.set_xlabel('Number of test songs');ax.legend(title='Cluster',ncol=5,fontsize=8);savefig('rebuilt_cluster_sizes')
 fig,ax=plt.subplots(figsize=(7.8,3.0),layout='constrained')
 for n in ['audio_pca','lyrics_lsa','audio_vae_b01','hybrid_vae','multimodal_vae']:
  q=d[(d.method==n)&(d.split=='test')&(d.protocol=='inductive_kmeans')].groupby('k_requested').ari.mean();ax.plot(q.index,q.values,marker='o',label=LABELS[n])
 ax.set_xlabel('Requested clusters');ax.set_ylabel('Test ARI');ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.2);savefig('rebuilt_k_sensitivity')
 ck=torch.load(P/'audio_vae_b01/11/model.pt',map_location='cpu',weights_only=True);model=MusicVAE(**ck['model_kwargs']);model.load_state_dict(ck['state_dict']);model.eval()
 z=np.load(P/'audio_vae_b01/11/representations.npz')['mu'];t=np.linspace(0,1,7);zi=np.stack([(1-v)*z[sample[0]]+v*z[sample[-1]] for v in t])
 with torch.no_grad():rec=model.decode(torch.from_numpy(zi).float())[0].numpy()
 fig,axes=plt.subplots(1,7,figsize=(8.2,1.4),layout='constrained')
 for ax,v,r in zip(axes,t,rec):ax.imshow(r[0],origin='lower',aspect='auto',vmin=-3,vmax=3,cmap='magma');ax.set_title(f't={v:.2f}');ax.set_xticks([]);ax.set_yticks([])
 savefig('rebuilt_interpolation')
 write_json(OUT/'report_manifest.json',{'script_sha256':sha256(__file__),'primary_signature':manifest['signature'],
  'metric_files':{str(p.relative_to(ROOT)):sha256(p) for p in [P/'metrics.csv',E/'metrics.csv']},'bootstrap_resamples':1000,'bootstrap_seed':20261029})
 print('Built tables, uncertainty analyses and 8 figure sets.',flush=True)
 print(pd.DataFrame(boot).to_string(index=False),flush=True)
if __name__=='__main__':main()
