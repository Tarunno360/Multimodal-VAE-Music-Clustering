"""Fetch a fixed small research sample from pinned public dataset revisions.

Raw recordings stay local. BanglaBeats' original song IDs and license are not
specified in this release; this sample is descriptive, not a track-disjoint test.
No downloaded Python code is executed.
"""
import json,hashlib,zipfile,subprocess,sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen, Request
from inspect_remote_zip import HTTPRangeFile
OUT=Path('data/bilingual');OUT.mkdir(parents=True,exist_ok=True)
SOURCES=Path('data/external_sources')
EN_REV='de188c963fd4539bc769b3feb83582e5a9595e36'
BN_REV='f3c4cd1e6172ac1bbdbb8222f3990f53f9ec7f8d'
def fetch(url,path):
 if not path.exists():
  with urlopen(Request(url,headers={'User-Agent':'music-vae-research/0.1'}),timeout=120) as r:
   data=r.read(30*1024*1024+1)
  if len(data)>30*1024*1024:raise RuntimeError('Unexpectedly large source file; refusing to save a partial download')
  path.write_bytes(data)
 return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 rows=[]
 SOURCES.mkdir(parents=True,exist_ok=True)
 metadata=SOURCES/'jamendolyrics_subsets_en_metadata.jsonl'
 if not metadata.exists():
  fetch('https://huggingface.co/datasets/jamendolyrics/jamendolyrics/resolve/'+EN_REV+'/subsets/en/metadata.jsonl',metadata)
 if not (SOURCES/'banglabeats_zip_inventory.json').exists():
  subprocess.run([sys.executable,str(Path(__file__).with_name('inspect_remote_zip.py'))],check=True)
 english=[json.loads(s) for s in (SOURCES/'jamendolyrics_subsets_en_metadata.jsonl').read_text().splitlines()]
 for i,r in enumerate(sorted(english,key=lambda x:x['name'])):
  # Paths in language-local metadata are relative to subsets/en/.
  remote='subsets/en/'+r['file_name']
  url='https://huggingface.co/datasets/jamendolyrics/jamendolyrics/resolve/'+EN_REV+'/'+quote(remote,safe='/')
  path=OUT/f'en_{i:02d}.mp3'
  digest=fetch(url,path)
  rows.append({'id':f'en_{i:02d}','audio_path':str(path),'language':'en','genre':r['genre'],
   'title':r['title'],'artist':r['artist'],'source':'JamendoLyrics','revision':EN_REV,
   'source_file':remote,'source_url':url,'license':r['license_type'],'sha256':digest,
   'offset_seconds':float(r['lines'][0]['start']),'duration_seconds':3.,'track_id':r['name']})
  print('Downloaded',path,flush=True)
 inventory=json.loads((SOURCES/'banglabeats_zip_inventory.json').read_text())
 genres=sorted({r['name'].split('/')[-2] for r in inventory})
 selected=[]
 for gi,genre in enumerate(genres):
  files=sorted([r for r in inventory if r['name'].split('/')[-2]==genre],key=lambda x:int(Path(x['name']).stem))
  n=3 if gi<4 else 2
  selected += [files[round(j*(len(files)-1)/(n-1))] for j in range(n)]
 url='https://huggingface.co/datasets/thisisjibon/banglabeats/resolve/'+BN_REV+'/data/data.zip'
 stream=HTTPRangeFile(url,4098785436)
 with zipfile.ZipFile(stream) as z:
  for i,r in enumerate(selected):
   path=OUT/f'bn_{i:02d}.wav'
   if not path.exists():path.write_bytes(z.read(r['name']))
   rows.append({'id':f'bn_{i:02d}','audio_path':str(path),'language':'bn','genre':r['name'].split('/')[-2],
    'title':'Unspecified in source','artist':'Unspecified in source','source':'BanglaBeats','revision':BN_REV,
    'source_file':r['name'],'source_url':url,'license':'Unspecified in source release',
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'offset_seconds':0.,'duration_seconds':3.,'track_id':None})
   print('Downloaded',path,r['name'],flush=True)
 (OUT/'manifest.json').write_text(json.dumps({'selection':'All 20 English tracks; 20 Bengali clips spaced over numeric file ranks within eight genres. Not selected by model scores.',
  'scope':'Audio-only bilingual feasibility cohort. Language is perfectly confounded with source. Bangla track identities and licensing unresolved. No redistribution of recordings.',
  'rows':rows},indent=2)+'\n')
 print('Complete',len(rows),'recordings; Bangla archive bytes read',stream.received,flush=True)
if __name__=='__main__':main()
