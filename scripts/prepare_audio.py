"""Deterministic new audio preprocessing, NOT a reconstruction of the legacy recipe."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import librosa

def spectrogram(path,offset=0.,duration=3.):
 y,sr=librosa.load(path,sr=22050,mono=True,offset=offset,duration=duration)
 if not len(y) or not np.isfinite(y).all():raise ValueError(f'Invalid waveform: {path}')
 y=librosa.util.fix_length(y,size=int(round(duration*sr)))
 if np.max(np.abs(y))<1e-8:raise ValueError(f'Silent waveform: {path}')
 power=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=2048,hop_length=512,n_mels=128,fmin=0,fmax=sr/2,power=2.,center=True)
 db=librosa.power_to_db(power,ref=np.max,top_db=80.)
 # Linear interpolation standardizes the time-axis length; it does not add information.
 grid=np.linspace(0,db.shape[1]-1,256)
 db=np.stack([np.interp(grid,np.arange(db.shape[1]),row) for row in db])
 return (np.clip(db,-80,0)/40.+1).astype('float32')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);args=p.parse_args()
 manifest=json.loads(Path(args.manifest).read_text()); arrays=[]
 for row in manifest['rows']:
  path=Path(row['audio_path'])
  if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(f'Hash mismatch: {path}')
  arrays.append(spectrogram(path,row.get('offset_seconds',0),row.get('duration_seconds',3)))
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 np.save(out/'spectrograms.npy',np.stack(arrays));np.save(out/'ids.npy',np.array([r['id'] for r in manifest['rows']]))
 recipe={'sample_rate':22050,'mono':True,'n_fft':2048,'hop_length':512,'n_mels':128,'power':2,'db_reference':'per-clip maximum','top_db':80,
 'time_frames':256,'time_resampling':'linear interpolation','output_range':[-1,1],'librosa':librosa.__version__,
 'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (out/'preprocessing.json').write_text(json.dumps(recipe,indent=2)+'\n');print('Prepared',len(arrays),'clips',flush=True)
if __name__=='__main__':main()
