"""Compare every locally tracked file with a fresh anonymous GitHub clone.

Run from this repository after committing and pushing. The local evidence file
is ignored by Git to avoid an impossible self-referential hash manifest.
"""
import argparse,datetime,hashlib,json,subprocess,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--repository',default='n-irei/instacart-repurchase-analysis');a=p.parse_args()
    root=Path(__file__).resolve().parent;work=a.workspace.resolve();work.mkdir(parents=True,exist_ok=True)
    assert '..' not in a.repository and len(a.repository.split('/'))==2
    def git(*args,cwd=root):return subprocess.check_output(['git','-C',str(cwd),*args])
    status=git('status','--porcelain').decode();assert not status,status
    api='https://api.github.com/repos/'+a.repository
    req=urllib.request.Request(api,headers={'User-Agent':'Instacart publication verification'})
    with urllib.request.urlopen(req,timeout=60) as r:info=json.load(r)
    assert info['private'] is False
    now=datetime.datetime.now(datetime.timezone.utc)
    checkout=work/('remote-'+now.strftime('%Y%m%dT%H%M%S'))
    assert checkout.resolve().parent==work
    subprocess.run(['git','clone','--bare','--depth','1','https://github.com/'+a.repository+'.git',str(checkout)],check=True)
    local=git('rev-parse','HEAD').decode().strip();remote=git('rev-parse','HEAD',cwd=checkout).decode().strip();assert local==remote
    names=git('ls-tree','-r','--name-only','HEAD').decode().splitlines()
    remote_names=git('ls-tree','-r','--name-only','HEAD',cwd=checkout).decode().splitlines();assert names==remote_names
    files=[]
    for name in names:
        raw=(root/name).read_bytes();committed=git('show',f'HEAD:{name}');public=git('show',f'HEAD:{name}',cwd=checkout)
        assert raw==committed==public,name
        files.append({'path':name,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'bytes_equal':True})
    report={'repository':'https://github.com/'+a.repository,'visibility':'PUBLIC','verified_utc':now.isoformat(),'unauthenticated_api_and_clone':True,'local_head':local,'remote_head':remote,'working_tree_clean':True,'all_tracked_files_byte_equal':True,'file_count':len(files),'files':files,'scope':'Every tracked file, including submission.csv. This local-only evidence file is intentionally ignored by Git.'}
    (root/'PUBLICATION_VERIFICATION.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'PUBLIC; HEAD equal; {len(files)} files SHA-256 and byte equality PASS')

if __name__=='__main__':main()
