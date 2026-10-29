"""Execute the results walkthrough without retraining models."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'tmp/jupyter'))
os.environ.setdefault('IPYTHONDIR',str(ROOT/'tmp/ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import nbformat
from nbclient import NotebookClient

def main():
 p=ROOT/'notebooks/Complete_Project.ipynb'
 notebook=nbformat.read(p,as_version=4)
 NotebookClient(notebook,timeout=120,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}}).execute()
 nbformat.validate(notebook)
 assert not any(o.get('output_type')=='error' for c in notebook.cells
                for o in c.get('outputs',[]))
 nbformat.write(notebook,p)
 print('Executed and validated',sum(c.cell_type=='code' for c in notebook.cells),'notebook cells.')
if __name__=='__main__':main()
