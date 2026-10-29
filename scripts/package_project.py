"""Build a reviewable source/results archive using an explicit inclusion list."""
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]

def main():
 files=set()
 for pattern in ['*.md','requirements*.txt','pyproject.toml','.gitignore',
   'musicvae/*.py','configs/*.json','scripts/*.py','tests/*.py',
   'notebooks/Complete_Project.ipynb','Project Requirement/assignment_brief.md',
   'Final Writing Latex/*.tex','Final Writing Latex/*.sty','Final Writing Latex/*.bib','Final Writing Latex/*.pdf',
   'Final Writing Latex/generated/*.tex','Final Writing Latex/figures/rebuilt_*.pdf',
   'Final Writing Latex/figures/rebuilt_*.png','Final Writing Latex/figures/bilingual.*',
   'output/pdf/*.pdf','audit/*.md','audit/*.json','audit/legacy_recomputed.csv','audit/tests.log',
   'audit/run_sources/**/*.py','audit/run_sources/**/*.json',
   'data/bilingual/manifest.json','data/bilingual/prepared/preprocessing.json']:
  files.update(p for p in ROOT.glob(pattern) if p.is_file())
 for directory in ['rebuilt_v1','requirements_extension','bilingual','analysis','robustness_2027','robustness_2028']:
  for p in (ROOT/'results'/directory).rglob('*'):
   if p.is_file() and p.suffix in ['.csv','.json']:
    files.add(p)
 banned={'.mp3','.wav','.npy','.npz','.pt','.h5','.joblib','.jsonl'}
 assert not any(p.suffix in banned for p in files)
 out=ROOT/'output';out.mkdir(exist_ok=True)
 manifest={'scope':'Review candidate: code, numeric summaries, scientific figures and paper; not a public licensed data release.',
  'excluded':'Original data archives, full lyrics, raw recordings, cached features, weights and fitted transforms remain in the full local workspace.',
  'files':[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)]}
 manifest_path=out/'source_package_manifest.json'
 manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
 archive=out/'VAE_Project_Source_and_Results.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(files):z.write(p,Path('VAE_Project')/p.relative_to(ROOT))
  z.write(manifest_path,'VAE_Project/source_package_manifest.json')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for entry in manifest['files']:
   assert hashlib.sha256(z.read('VAE_Project/'+entry['path'])).hexdigest()==entry['sha256']
 (out/'SHA256SUMS.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
 print(f'Verified {len(files)} files; archive {archive.stat().st_size/1024**2:.1f} MiB: {archive}')

if __name__=='__main__':main()
