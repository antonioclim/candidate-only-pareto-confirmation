from pathlib import Path
import csv,hashlib
OFFICIAL_VERSION='2020-10-06';OFFICIAL_DOI='10.25832/time_series/2020-10-06'
OFFICIAL_FILENAME='opsd_time_series_60min_singleindex_2020-10-06.csv'
OFFICIAL_SHA256='6A7F2BC571314CBF9C321CC03437691CD4BE95C3A6F075E60FF99E8035C704C8'
def sha256_file(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest().upper()
def verify_official_source(path):
 p=Path(path)
 if not p.exists():return {'exists':False,'valid':False,'path':str(p),'reason':'file_missing'}
 d=sha256_file(p);return {'exists':True,'valid':d==OFFICIAL_SHA256,'path':str(p),'sha256':d,'expected_sha256':OFFICIAL_SHA256,'reason':'ok' if d==OFFICIAL_SHA256 else 'sha256_mismatch'}
def chronological_split(rows,fraction=.5):
 if not 0<fraction<1:raise ValueError
 z=sorted(rows,key=lambda r:r['utc_timestamp']);k=max(1,min(len(z)-1,int(len(z)*fraction)));return z[:k],z[k:]
