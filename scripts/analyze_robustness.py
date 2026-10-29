"""Descriptive follow-up on two additional grouped splits; never select a winner."""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from musicvae.data import sha256
from musicvae.experiment import write_json

METHODS = {'audio_pca':'Audio PCA', 'audio_statistics':'Audio statistics',
 'lyrics_lsa':'Lyrics LSA', 'fusion_pca':'Fusion PCA', 'audio_ae':'Audio AE',
 'audio_vae_b01':'Audio VAE (0.01)', 'conv_vae':'Conv VAE',
 'multimodal_vae':'Multimodal VAE', 'hybrid_vae':'Hybrid VAE',
 'multimodal_shuffled':'Shuffled text'}

def main():
 out = ROOT/'results/analysis'; tex = ROOT/'Final Writing Latex/generated'
 frames, records = [], []
 for seed in [2026, 2027, 2028]:
  p = ROOT/('results/rebuilt_v1' if seed == 2026 else f'results/robustness_{seed}')
  assert json.loads((p/'completion.json').read_text())['completed']
  d = pd.read_csv(p/'metrics.csv')
  q = d[(d.method.isin(METHODS)) & (d.split=='test') &
        (d.protocol=='inductive_kmeans') & (d.k_requested==5)].copy()
  assert len(q)==50
  q['split_seed'] = seed; frames.append(q)
  m = json.loads((p/'manifest.json').read_text())
  records.append({'split_seed':seed, 'split_sizes':m['split_sizes'],
   'signature':m['signature'], 'metrics_sha256':sha256(p/'metrics.csv')})
 d = pd.concat(frames, ignore_index=True)
 d.to_csv(out/'split_sensitivity_runs.csv', index=False)
 summary = d.groupby(['method','split_seed']).ari.agg(['mean','std'])
 summary.to_csv(out/'split_sensitivity_summary.csv')
 means = summary['mean'].unstack()
 contrasts = []
 for a,b in [('hybrid_vae','audio_vae_b01'), ('hybrid_vae','lyrics_lsa'),
             ('multimodal_vae','multimodal_shuffled')]:
  contrasts.append({'a':a, 'b':b, 'differences':{
   str(s):float(means.loc[a,s]-means.loc[b,s]) for s in [2026,2027,2028]}})
 write_json(out/'split_sensitivity.json', {'splits':records,'contrasts':contrasts,
  'followup_after_primary_analysis':True,'hyperparameters_retuned':False,
  'interpretation':'Overlapping grouped splits of one corpus; descriptive, not independent datasets.'})
 rows=[]
 for method,label in METHODS.items():
  vals = [f'${summary.loc[(method,s),"mean"]:.3f}\\pm{summary.loc[(method,s),"std"]:.3f}$' for s in [2026,2027,2028]]
  rows.append(' & '.join([label]+vals)+r' \\')
 (tex/'robustness_table.tex').write_text(
  '\\begin{table}[htbp]\n\\centering\n\\small\n'
  '\\caption{Split sensitivity: held-out K-means ARI at $k=5$, mean $\\pm$ sample SD over five training seeds per split. Split 2026 is primary; 2027 and 2028 are follow-up evaluations with unchanged settings. These are overlapping splits of the same corpus.}\n'
  '\\label{tab:robustness}\n\\begin{tabular}{lrrr}\n\\toprule\n'
  'Representation & Split 2026 & Split 2027 & Split 2028 \\\\\n\\midrule\n'+
  '\n'.join(rows)+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
 def sequence(a,b):
  return ', '.join(f'{means.loc[a,s]-means.loc[b,s]:+.3f}' for s in [2026,2027,2028])
 ordered=sum(means.loc['lyrics_lsa',s]>means.loc['hybrid_vae',s]>means.loc['audio_vae_b01',s] for s in [2026,2027,2028])
 (tex/'robustness_abstract.tex').write_text(
  'Two additional grouped splits preserve this ordering.\n' if ordered==3 else
  'Two additional grouped splits assess the sensitivity of this comparison.\n')
 (tex/'robustness_narrative.tex').write_text(
  'Two follow-up group-disjoint splits (seeds 2027 and 2028) repeat six neural models and four baselines with the same five training seeds and unchanged hyperparameters (Table~\\ref{tab:robustness}). '
  f'The hybrid-minus-audio-VAE mean ARI differences across splits 2026, 2027 and 2028 are {sequence("hybrid_vae","audio_vae_b01")}; hybrid-minus-text differences are {sequence("hybrid_vae","lyrics_lsa")}. '
  'These follow-up evaluations assess split sensitivity after inspecting the primary results; they are not independent dataset replications or confirmatory tests.\n')
 seconds=sum(json.loads(p.read_text())['seconds'] for s in [2027,2028]
             for p in (ROOT/f'results/robustness_{s}').glob('*/*/training.json'))
 (tex/'robustness_compute.tex').write_text(
  f'The additional split-sensitivity study trained 60 neural models in {seconds/60:.1f} minutes of summed training wall time; preprocessing and evaluation are additional.\n')
 print(summary.to_string())
 print(json.dumps(contrasts,indent=2))

if __name__=='__main__':main()
