"""Build named and anonymous research drafts; preserve editable source files."""
from pathlib import Path
import os,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'Final Writing Latex';OUT=ROOT/'output/pdf'
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 tectonic=shutil.which('tectonic')
 env=os.environ.copy()
 cache=ROOT/'tmp/tectonic'
 if tectonic and cache.exists():env['TECTONIC_CACHE_DIR']=str(cache)
 for source,target in [('neurips_2024','VAE_Music_Clustering_Report'),('anonymous','VAE_Music_Clustering_Anonymous_Draft')]:
  if tectonic:
   command=[tectonic,'-X','compile','--keep-logs','--keep-intermediates',source+'.tex']
   if cache.exists():command.insert(-1,'--only-cached')
   subprocess.run(command,cwd=PAPER,env=env,check=True)
  else:
   pdflatex=shutil.which('pdflatex');bibtex=shutil.which('bibtex')
   if not pdflatex or not bibtex:raise SystemExit('Install Tectonic or a complete TeX Live distribution, then retry.')
   subprocess.run([pdflatex,'-interaction=nonstopmode','-halt-on-error',source+'.tex'],cwd=PAPER,check=True)
   subprocess.run([bibtex,source],cwd=PAPER,check=True)
   for _ in range(2):subprocess.run([pdflatex,'-interaction=nonstopmode','-halt-on-error',source+'.tex'],cwd=PAPER,check=True)
  shutil.copy2(PAPER/(source+'.pdf'),OUT/(target+'.pdf'))
 print('Built named and anonymous PDFs in',OUT)
if __name__=='__main__':main()
