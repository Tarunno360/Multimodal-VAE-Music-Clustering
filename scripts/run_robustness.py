"""Follow-up split sensitivity after the primary study; no hyperparameters retuned."""
from pathlib import Path
import sys,subprocess,os,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 for split_seed in [2027,2028]:
  out=ROOT/f'results/robustness_{split_seed}';out.mkdir(parents=True,exist_ok=True)
  if (out/'completion.json').exists():continue
  cmd=[sys.executable,'-m','musicvae.experiment','--config',f'configs/robustness_{split_seed}.json','--output',f'results/robustness_{split_seed}']
  if (out/'manifest.json').exists():cmd.append('--resume')
  with (out/'run.log').open('a') as f:subprocess.run(cmd,cwd=ROOT,env={**os.environ,'LOKY_MAX_CPU_COUNT':'4'},stdout=f,stderr=subprocess.STDOUT,check=True)
  print('Completed split',split_seed,flush=True)
if __name__=='__main__':main()
