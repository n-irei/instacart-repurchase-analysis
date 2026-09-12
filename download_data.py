"""Fetch the user-approved third-party Kaggle mirror, pinned to version 1."""
import argparse,datetime,hashlib,json,urllib.request,zipfile
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    a.destination.mkdir(parents=True,exist_ok=True)
    url='https://www.kaggle.com/api/v1/datasets/download/psparks/instacart-market-basket-analysis?datasetVersionNumber=1'
    dest=a.destination/'instacart-psparks-v1.zip'
    if dest.exists():raise FileExistsError(f'Refusing to overwrite {dest}')
    req=urllib.request.Request(url,headers={'User-Agent':'Instacart reproducible analysis'})
    with urllib.request.urlopen(req,timeout=90) as response,dest.open('wb') as output:
        for block in iter(lambda:response.read(1024*1024),b''):output.write(block)
    data=a.destination/'data';data.mkdir(exist_ok=True)
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for n in z.namelist():assert Path(n).name==n and n.endswith('.csv')
        z.extractall(data)
    manifest={'source_url':url,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_sha256':sha(dest),'official_byte_equality_verified':False,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(data.glob('*.csv'))}}
    (a.destination/'download_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Downloaded {dest.stat().st_size:,} bytes; CSVs in {data}')

if __name__=='__main__':main()
