"""Download public dataset inventories only; never execute remote dataset scripts."""
import json
from pathlib import Path
from urllib.request import urlopen, Request
out=Path('data/external_sources');out.mkdir(parents=True,exist_ok=True)
for repo in ['thisisjibon/banglabeats','jamendolyrics/jamendolyrics']:
 url='https://huggingface.co/api/datasets/'+repo+'/tree/main'
 with urlopen(Request(url,headers={'User-Agent':'music-vae-research/0.1'}),timeout=60) as r: data=json.load(r)
 (out/(repo.split('/')[-1]+'_tree.json')).write_text(json.dumps(data,indent=2))
 print(repo,[(r['path'],r.get('size'),r['type']) for r in data][:60])
 with urlopen('https://huggingface.co/api/datasets/'+repo,timeout=60) as r: info=json.load(r)
 (out/(repo.split('/')[-1]+'_info.json')).write_text(json.dumps(info,indent=2))
 print('revision',info.get('sha'),'card',info.get('cardData'))
