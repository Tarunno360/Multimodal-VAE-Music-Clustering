"""Map the existing repository's verified inputs into the frozen run layout.

This establishes byte identity, not the original recordings' provenance or rights.
Existing destination files must match; nothing is silently overwritten.
"""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1]
SOURCE_PATHS={
 'spectrograms':'dataset/audio/processed_spectrograms.npy',
 'ids':'dataset/audio/processed_ids.npy',
 'metadata':'dataset/audio/dataset.csv',
 'cached_lyrics':'dataset/lyrics/lyrics_embeddings.npy',
}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=ROOT,help='Root of the original repository checkout')
 args=parser.parse_args()
 inputs=json.loads((ROOT/'audit/data_inventory.json').read_text())['manifest']
 work=[]
 for key,relative in SOURCE_PATHS.items():
  source=args.source/relative;target=ROOT/inputs[key]['path'];expected=inputs[key]['sha256']
  if not source.is_file():raise FileNotFoundError(f'Missing existing repository input: {source}')
  if digest(source)!=expected:raise ValueError(f'Input differs from the frozen experiment: {source}')
  if target.exists() and digest(target)!=expected:raise ValueError(f'Refusing to overwrite a different file: {target}')
  work.append((source,target,expected))
 for source,target,expected in work:
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  assert digest(target)==expected
  print('Verified',target.relative_to(ROOT))

if __name__=='__main__':main()
