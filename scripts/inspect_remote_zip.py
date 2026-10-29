import io,json,zipfile
from urllib.request import Request,urlopen
from pathlib import Path
class HTTPRangeFile(io.RawIOBase):
 def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.received=0
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  return self.pos
 def read(self,n=-1):
  n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
  if n<=0:return b''
  req=Request(self.url,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}','User-Agent':'music-vae-research/0.1'})
  with urlopen(req,timeout=90) as r:
   if r.status!=206:raise RuntimeError(f'Server did not honor byte range: {r.status}; refusing full archive')
   expected=f'bytes {self.pos}-{self.pos+n-1}/'
   if not r.headers.get('Content-Range','').startswith(expected):raise RuntimeError('Incorrect Content-Range '+str(r.headers))
   data=r.read(n+1)
  if len(data)!=n:raise IOError('Range length mismatch')
  self.pos+=n;self.received+=n
  return data
if __name__=='__main__':
 f=HTTPRangeFile('https://huggingface.co/datasets/thisisjibon/banglabeats/resolve/f3c4cd1e6172ac1bbdbb8222f3990f53f9ec7f8d/data/data.zip',4098785436)
 with zipfile.ZipFile(f) as z:
  rows=[{'name':i.filename,'size':i.file_size,'compressed':i.compress_size} for i in z.infolist() if i.filename.endswith('.wav') and '__MACOSX' not in i.filename and not Path(i.filename).name.startswith('._')]
  Path('data/external_sources/banglabeats_zip_inventory.json').write_text(json.dumps(rows,indent=2))
  print('Audio files',len(rows),'Bytes read',f.received,'First files',rows[:25])
