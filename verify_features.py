"""Independent raw-data recomputation of sampled candidate feature vectors.

This check scans prior in chunks, never reads train labels, and reconstructs
features for two users from each partition from their prior orders alone.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from analysis import NUMERIC

def main(data,private,out):
    picked=[];stored=[]
    for part in ['train','tune','hold','test']:
        f=pd.read_parquet(private/f'{part}.parquet')
        users=np.sort(f.user_id.unique())[[0,-1]]
        picked.extend(users.tolist());stored.append(f[f.user_id.isin(users)])
    stored=pd.concat(stored).set_index(['user_id','product_id'])
    orders=pd.read_csv(data/'orders.csv');orders=orders[orders.user_id.isin(picked)].copy()
    ids=set(orders.loc[orders.eval_set.eq('prior'),'order_id'])
    chunks=[]
    for c in pd.read_csv(data/'order_products__prior.csv',chunksize=1000000):
        chunks.append(c[c.order_id.isin(ids)])
    purchases=pd.concat(chunks).merge(orders,on='order_id',validate='many_to_one')
    tested=0;max_abs=0.
    for uid in picked:
        oo=orders[orders.user_id.eq(uid)].sort_values('order_number')
        hist=oo[oo.eval_set.eq('prior')];target=oo[oo.eval_set.ne('prior')].iloc[0]
        pp=purchases[purchases.user_id.eq(uid)]
        assert set(pp.product_id)==set(stored.loc[uid].index)
        sizes=pp.groupby('order_id').size();n=len(hist)
        day=dict(zip(oo.order_number,oo.days_since_prior_order.fillna(0).cumsum()))
        for pid in sorted(pp.product_id.unique()):
            p=pp[pp.product_id.eq(pid)].sort_values('order_number');first=p.order_number.min();last=p.order_number.max();count=len(p)
            expected={'frequency':count/n,'frequency_since_first':count/(n-first+1),'orders_since_last':target.order_number-last,'last3_frequency':p.order_number.gt(n-3).sum()/min(n,3),'purchase_count':count,'mean_cart_position':p.add_to_cart_order.mean(),'mean_purchase_interval':(last-first)/max(count-1,1),'history_orders':n,'mean_basket_size':sizes.mean(),'basket_size_std':sizes.std(),'unique_products':pp.product_id.nunique(),'prior_reorder_rate':pp.reordered.mean(),'mean_order_gap':hist.days_since_prior_order.mean(),'target_gap':target.days_since_prior_order,'target_dow':target.order_dow,'target_hour':target.order_hour_of_day,'days_since_product':day[target.order_number]-day[last],'last_basket_size':sizes.loc[hist.iloc[-1].order_id]}
            a=np.array([expected[c] for c in NUMERIC]);b=stored.loc[(uid,pid),NUMERIC].to_numpy(dtype=float)
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
            max_abs=max(max_abs,float(np.max(np.abs(a-b))));tested+=1
    result={'pass':True,'sampled_users':len(picked),'sampled_candidate_rows':tested,'numeric_features':len(NUMERIC),'maximum_absolute_difference':max_abs,'relative_tolerance':2e-6,'labels_file_read':False,'prior_chunk_rows':1000000,'scope':'Independent source recomputation for 2 users from each of four partitions; complements full-table audits, not a proof for every feature row'}
    (out/'independent_feature_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();main(a.data,a.private,a.output)
