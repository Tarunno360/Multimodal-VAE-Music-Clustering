"""Small transductive English/Bangla course demonstration, not a language benchmark."""
from pathlib import Path
import os,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from musicvae.experiment import train_model,write_json
from musicvae.metrics import evaluate
from musicvae.data import sha256
from build_report import table

def main():
 out=ROOT/'results/bilingual';out.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((ROOT/'data/bilingual/manifest.json').read_text());rows=manifest['rows']
 raw=np.load(ROOT/'data/bilingual/prepared/spectrograms.npy');n=len(raw);assert n==40
 y=np.array([0 if r['language']=='en' else 1 for r in rows]);idx=np.arange(n)
 tr,va=train_test_split(idx,test_size=.2,random_state=2026,stratify=y)
 splits={'train':np.sort(tr),'val':np.sort(va),'all':idx}
 a=raw.reshape(n,32,4,64,4).mean((2,4));mean=float(a[tr].mean());std=float(a[tr].std());a=((a-mean)/std).astype('float32')[:,None]
 dummy=np.zeros((n,1),dtype='float32');condition=np.eye(2,dtype='float32')[y]
 config={'seeds':[11,22,33,44,55],'epochs':100,'patience':15,'warmup':20,'batch_size':16,'learning_rate':.001,'latent_dim':8,'threads':4}
 torch.set_num_threads(4);metric_rows=[];embeddings={};preds={}
 with threadpool_limits(limits=4):
  pca=PCA(n_components=8,svd_solver='full').fit(a[tr].reshape(len(tr),-1));zp=pca.transform(a.reshape(n,-1))
 for seed in config['seeds']:
  spec={'name':'audio_vae','kind':'dense','beta':.01}
  z,_=train_model(config,spec,seed,a,dummy,condition,splits,out/'audio_vae'/str(seed))
  for name,rep in [('PCA',zp),('VAE',z)]:
   for k in [2,5]:
    with threadpool_limits(limits=4):cl=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(rep)
    metric_rows.append({'method':name,'seed':seed,'k':k,'protocol':'transductive_all_40','label_type':'language_equals_source',**evaluate(rep,cl,y)})
    if seed==11 and k==5:preds[name]=cl;embeddings[name]=rep
 pd.DataFrame(metric_rows).to_csv(out/'metrics.csv',index=False);write_json(out/'metrics.json',metric_rows)
 pd.DataFrame({'id':[r['id'] for r in rows],'language':[r['language'] for r in rows],'source':[r['source'] for r in rows],
 'genre':[r['genre'] for r in rows],'split':['train' if i in tr else 'val' for i in idx],**{k+'_cluster':v for k,v in preds.items()}}).to_csv(out/'assignments.csv',index=False)
 write_json(out/'manifest.json',{'config':config,'data_manifest_sha256':sha256(ROOT/'data/bilingual/manifest.json'),
 'script_sha256':sha256(__file__),'audio_mean':mean,'audio_std':std,'evaluation':'All 40 clips including training, descriptive only',
 'limitations':['Language perfectly confounded with dataset source','Bengali original track IDs missing','Bengali paired lyrics unavailable','License absent from Bengali release','Only 20 clips per language']})
 df=pd.DataFrame(metric_rows);q=df[df.k==5]
 def f(name,m):
  s=q[q.method==name][m];return f'${s.mean():.3f}\\pm{s.std():.3f}$'
 table('bilingual_table.tex','Small bilingual demonstration, $k=5$, mean $\\pm$ SD. K-means fits and scores all 40 clips, including training clips. ARI compares language, which equals source.','tab:bilingual',
 ['Representation','Silhouette','CH','Language ARI'],[[name,f(name,'silhouette'),f(name,'calinski_harabasz'),f(name,'ari')] for name in ['PCA','VAE']])
 tex=ROOT/'Final Writing Latex/generated'
 (tex/'bilingual_methods.tex').write_text('We obtain 20 explicitly English-labeled recordings from JamendoLyrics \\citep{durand2023} and 20 Bengali clips from BanglaBeats \\citep{jibon2023}. Repository revisions and file hashes are pinned. English excerpts start at the first annotated sung line and last three seconds; Bengali excerpts are the released three-second clips, selected at spaced numeric file ranks across eight genres. A shared new recipe uses mono 22,050-Hz audio, FFT size 2,048, hop 512, 128 mel bins, per-clip relative dB clipped at $-80$ dB, and linear time interpolation to 256 frames, followed by the same pooling. A fixed language-stratified split uses 32 clips for fitting and eight for validation, with an eight-dimensional VAE and PCA. This separate course demonstration fits and evaluates K-means on all 40 representations; it is explicitly transductive and makes no held-out performance claim. BanglaBeats does not provide original song IDs in this release, so distinct-song independence cannot be verified. It also provides no matched lyrics; the multimodal study remains on the supplied corpus.\n')
 (tex/'bilingual_results.tex').write_text('\\input{generated/bilingual_table.tex}\nBoth languages pass through a common audio preprocessing and clustering implementation. Table~\\ref{tab:bilingual} reports internal scores and language agreement for all clips. Language is perfectly confounded with dataset source and may also covary with genre, recording practice and excerpt selection. These numbers therefore demonstrate execution on bilingual inputs; they do not establish language recognition, language-invariant music structure or generalization to new songs. The appendix plots actual language counts in each cluster.\n')
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(8,5.3),layout='constrained')
 for j,name in enumerate(['PCA','VAE']):
  with threadpool_limits(limits=4):xy=TSNE(n_components=2,perplexity=10,random_state=11,init='pca',learning_rate='auto').fit_transform(embeddings[name])
  for v,lang in [(0,'English / JamendoLyrics'),(1,'Bangla / BanglaBeats')]:axes[0,j].scatter(xy[y==v,0],xy[y==v,1],label=lang,s=26)
  axes[0,j].set_title(name+' t-SNE (all clips)');axes[0,j].set_xticks([]);axes[0,j].set_yticks([])
  counts=np.array([[(preds[name][y==v]==k).sum() for k in range(5)] for v in [0,1]])
  axes[1,j].bar(range(5),counts[0],label='English');axes[1,j].bar(range(5),counts[1],bottom=counts[0],label='Bangla')
  axes[1,j].set_xlabel('Cluster');axes[1,j].set_ylabel('Clips');axes[1,j].set_xticks(range(5))
 axes[0,0].legend(fontsize=6);axes[1,0].legend(fontsize=8)
 for ext in ['pdf','png']:plt.savefig(ROOT/f'Final Writing Latex/figures/bilingual.{ext}',bbox_inches='tight',dpi=180)
 plt.close()
 (tex/'bilingual_figure.tex').write_text('\\begin{figure}[h]\n\\centering\\includegraphics[width=\\linewidth]{figures/bilingual.pdf}\n\\caption{Bilingual course demonstration: t-SNE colored by annotated language/source and actual language counts in the five seed-11 clusters. All 40 clips are included. Bengali original song IDs are unavailable, and language and source cannot be disentangled.}\n\\end{figure}\n')
 write_json(out/'completion.json',{'completed':True,'n_clips':40,'seeds':config['seeds']})
 print(q.groupby('method')[['silhouette','calinski_harabasz','ari']].mean().to_string())
if __name__=='__main__':main()
