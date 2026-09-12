"""Independent Instacart baseline and user-disjoint holdout pipeline.

Usage: python analysis.py --data ../work/data --private ../work/private
Raw data, intermediate features, labels and models stay under --private.
"""
import argparse, gc, hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, log_loss
import lightgbm as lgb
from metrics import grouped_f1

SEED=20260912
NUMERIC=['frequency','frequency_since_first','orders_since_last','last3_frequency',
         'purchase_count','mean_cart_position','mean_purchase_interval','history_orders',
         'mean_basket_size','basket_size_std','unique_products','prior_reorder_rate',
         'mean_order_gap','target_gap','target_dow','target_hour','days_since_product',
         'last_basket_size']
CATEGORICAL=['aisle_id','department_id']

def write_json(path,obj):
    def convert(x):
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=convert),encoding='utf-8')

def log(msg): print(time.strftime('%H:%M:%S'),msg,flush=True)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def prepare(data,private,out):
    audit={}; eda={}
    def check(name,condition,detail=None):
        audit[name]={'pass':bool(condition),'detail':detail}
        if not condition:
            write_json(out/'join_leakage_audit.json',audit)
            raise AssertionError(name)
    log('Reading independent inputs')
    provenance=out/'data_provenance.json'
    if provenance.exists():
        for name,info in json.loads(provenance.read_text(encoding='utf-8'))['files'].items():
            check('sha256_'+name,digest(data/name)==info['sha256'])
    orders=pd.read_csv(data/'orders.csv',dtype={'order_id':'int32','user_id':'int32','eval_set':'category','order_number':'int16','order_dow':'int8','order_hour_of_day':'int8','days_since_prior_order':'float32'})
    products=pd.read_csv(data/'products.csv',dtype={'product_id':'int32','aisle_id':'int16','department_id':'int8'})
    aisles=pd.read_csv(data/'aisles.csv'); departments=pd.read_csv(data/'departments.csv')
    opd={'order_id':'int32','product_id':'int32','add_to_cart_order':'int16','reordered':'int8'}
    prior=pd.read_csv(data/'order_products__prior.csv',dtype=opd)
    labels=pd.read_csv(data/'order_products__train.csv',dtype=opd)
    frames={'orders':orders,'products':products,'aisles':aisles,'departments':departments,'order_products__prior':prior,'order_products__train':labels}
    eda['tables']={k:{'rows':len(v),'columns':len(v.columns),'nulls':v.isna().sum().to_dict(),'memory_bytes':int(v.memory_usage(deep=True).sum())} for k,v in frames.items()}
    for name,key in [('orders','order_id'),('products','product_id'),('aisles','aisle_id'),('departments','department_id')]:
        f=frames[name]; check(name+'_unique_key',f[key].notna().all() and f[key].is_unique)
    for name in ['order_products__prior','order_products__train']:
        f=frames[name]
        check(name+'_unique_order_product',not f.duplicated(['order_id','product_id']).any())
        check(name+'_nonnull',not f.isna().any().any())
        check(name+'_binary_reordered',f.reordered.isin([0,1]).all())
        check(name+'_product_foreign_key',f.product_id.isin(products.product_id).all())
    check('product_aisle_fk',products.aisle_id.isin(aisles.aisle_id).all())
    check('product_department_fk',products.department_id.isin(departments.department_id).all())
    metadata=products.merge(aisles,on='aisle_id',validate='many_to_one').merge(departments,on='department_id',validate='many_to_one')
    check('metadata_row_count',len(metadata)==len(products))
    check('null_gap_exactly_first_order',np.array_equal(orders.days_since_prior_order.isna(),orders.order_number.eq(1)))
    check('no_other_orders_nulls',not orders.drop(columns='days_since_prior_order').isna().any().any())
    check('valid_order_domain',orders.order_dow.between(0,6).all() and orders.order_hour_of_day.between(0,23).all() and orders.days_since_prior_order.dropna().between(0,30).all())
    check('valid_eval_sets',set(orders.eval_set)=={'prior','train','test'})
    orders=orders.sort_values(['user_id','order_number'])
    check('consecutive_user_orders',orders.order_number.eq(orders.groupby('user_id').cumcount()+1).all())
    target=orders[orders.eval_set.ne('prior')].copy()
    check('one_final_target_per_user',target.user_id.is_unique and len(target)==orders.user_id.nunique())
    maxnum=orders.groupby('user_id').order_number.max()
    check('target_is_last_order',np.array_equal(target.order_number, target.user_id.map(maxnum)))
    known=target[target.eval_set.eq('train')].user_id.to_numpy()
    train_users,remainder=train_test_split(known,test_size=.4,random_state=SEED)
    tune_users,hold_users=train_test_split(remainder,test_size=.5,random_state=SEED+1)
    split=np.full(int(orders.user_id.max())+1,'test',dtype='U5')
    split[train_users]='train';split[tune_users]='tune';split[hold_users]='hold'
    target['split']=split[target.user_id]
    splitsets={k:set(target.loc[target.split.eq(k),'user_id']) for k in ['train','tune','hold','test']}
    check('pairwise_disjoint_users',all(not splitsets[a]&splitsets[b] for i,a in enumerate(splitsets) for b in list(splitsets)[i+1:]))
    audit['split_counts']={k:len(v) for k,v in splitsets.items()}
    target[['user_id','order_id','split']].to_csv(private/'user_split.csv',index=False)
    audit['split_manifest_sha256']=digest(private/'user_split.csv')
    po=orders[orders.eval_set.eq('prior')].copy()
    check('prior_order_fk_and_membership',prior.order_id.isin(po.order_id).all())
    check('train_label_order_fk_and_membership',labels.order_id.isin(target.loc[target.eval_set.eq('train'),'order_id']).all())
    check('all_prior_orders_have_items',po.order_id.isin(prior.order_id).all())
    check('all_train_orders_have_items',target.loc[target.eval_set.eq('train'),'order_id'].isin(labels.order_id).all())
    eda['eval_set_orders']=orders.eval_set.value_counts().to_dict()
    eda['prior_orders_per_user']=po.groupby('user_id').size().describe(percentiles=[.1,.5,.9,.99]).to_dict()
    eda['prior_gap_counts']=po.days_since_prior_order.value_counts().sort_index().to_dict()
    eda['prior_hour_counts']=po.order_hour_of_day.value_counts().sort_index().to_dict()
    eda['prior_dow_counts']=po.order_dow.value_counts().sort_index().to_dict()
    eda['prior_row_reorder_rate']=float(prior.reordered.mean())
    counts=prior.groupby('order_id',sort=False).size()
    eda['prior_basket_size']=counts.describe(percentiles=[.1,.5,.9,.99]).to_dict()
    top=prior.product_id.value_counts().head(15).rename_axis('product_id').reset_index(name='prior_purchases').merge(metadata,on='product_id',validate='one_to_one')
    top.to_csv(out/'eda_top_products.csv',index=False)
    # Relative day origin: first order = 0. Its unknown preceding interval is excluded from gap averages.
    orders['relative_day']=orders.days_since_prior_order.fillna(0).groupby(orders.user_id).cumsum().astype('float32')
    po=orders[orders.eval_set.eq('prior')]
    lookup=po.set_index('order_id')
    log('Aggregating historical user-product candidates')
    for c in ['user_id','order_number','relative_day']:
        prior[c]=prior.order_id.map(lookup[c])
    check('historical_join_no_row_loss',len(prior)==eda['tables']['order_products__prior']['rows'] and not prior[['user_id','order_number','relative_day']].isna().any().any())
    hist=po.groupby('user_id').agg(history_orders=('order_number','max'),mean_order_gap=('days_since_prior_order','mean'))
    prior['recent3']=prior.order_number.gt(prior.user_id.map(hist.history_orders)-3).astype('int8')
    pairs=prior.groupby(['user_id','product_id'],sort=False,observed=True).agg(purchase_count=('order_id','size'),first_order=('order_number','min'),last_order=('order_number','max'),last_day=('relative_day','max'),mean_cart_position=('add_to_cart_order','mean'),last3_count=('recent3','sum'),repeat_sum=('reordered','sum')).reset_index()
    check('historical_reordered_counts',pairs.repeat_sum.eq(pairs.purchase_count-1).all())
    first_indices=prior.groupby(['user_id','product_id'],sort=False).order_number.idxmin()
    check('first_purchase_reordered_zero',prior.loc[first_indices,'reordered'].eq(0).all())
    del first_indices
    baskets=counts.rename('basket_size').reset_index();baskets['user_id']=baskets.order_id.map(lookup.user_id)
    user=baskets.groupby('user_id').basket_size.agg(mean_basket_size='mean',basket_size_std='std')
    user=user.join(hist).join(pairs.groupby('user_id').size().rename('unique_products'))
    user['prior_reorder_rate']=prior.groupby('user_id').reordered.mean()
    lastids=po.sort_values('order_number').groupby('user_id').tail(1).set_index('user_id').order_id
    user['last_basket_size']=lastids.map(counts)
    check('user_features_complete',not user.isna().any().any())
    eda['optimized_prior_working_bytes']=int(prior.memory_usage(deep=True).sum())
    eda['candidate_pairs_all']=len(pairs)
    del prior,baskets,counts;gc.collect()
    for c in ['purchase_count','first_order','last_order','last3_count']:pairs[c]=pairs[c].astype('int16')
    for c in ['last_day','mean_cart_position']:pairs[c]=pairs[c].astype('float32')
    pairs.drop(columns='repeat_sum',inplace=True)
    target=target.drop(columns='split').merge(orders[['order_id','relative_day']],on='order_id',validate='one_to_one')
    target['split']=split[target.user_id]
    # Only reordered=1 items are true competition labels; newly purchased products are excluded.
    positive=labels[labels.reordered.eq(1)][['order_id','product_id']]
    labelkeys=pd.MultiIndex.from_frame(positive)
    label_counts=positive.groupby('order_id').size()
    target['truth_count']=target.order_id.map(label_counts).fillna(0).astype('int16')
    # New item labels are not features. They are used only for a candidate-consistency audit.
    all_label_users=labels[['order_id','product_id','reordered']].merge(target[['order_id','user_id']],on='order_id',validate='many_to_one')
    pairkeys=pd.MultiIndex.from_frame(pairs[['user_id','product_id']])
    is_historical=pd.MultiIndex.from_frame(all_label_users[['user_id','product_id']]).isin(pairkeys)
    check('train_reordered_matches_history',np.array_equal(is_historical,all_label_users.reordered.eq(1)))
    del pairkeys,all_label_users,labels;gc.collect()
    for name in ['train','tune','hold','test']:
        tt=target[target.split.eq(name)].sort_values('order_id').reset_index(drop=True)
        tt['group']=np.arange(len(tt),dtype='int32')
        tt.to_parquet(private/f'{name}_orders.parquet',index=False)
        f=pairs[pairs.user_id.isin(tt.user_id)].merge(user.reset_index(),on='user_id',validate='many_to_one')
        f=f.merge(tt[['user_id','order_id','order_number','relative_day','days_since_prior_order','order_dow','order_hour_of_day','group']],on='user_id',validate='many_to_one')
        check(name+'_features_before_target',f.last_order.lt(f.order_number).all())
        check(name+'_history_target_adjacent',f.history_orders.eq(f.order_number-1).all())
        f['frequency']=f.purchase_count/f.history_orders
        f['frequency_since_first']=f.purchase_count/(f.history_orders-f.first_order+1)
        f['orders_since_last']=f.order_number-f.last_order
        f['last3_frequency']=f.last3_count/np.minimum(f.history_orders,3)
        # Single purchase has no observed recurrence interval: encode 0 and retain purchase_count.
        f['mean_purchase_interval']=(f.last_order-f.first_order)/np.maximum(f.purchase_count-1,1)
        f['days_since_product']=f.relative_day-f.last_day
        f=f.rename(columns={'days_since_prior_order':'target_gap','order_dow':'target_dow','order_hour_of_day':'target_hour'})
        f=f.merge(products[['product_id','aisle_id','department_id']],on='product_id',validate='many_to_one')
        f['label']=pd.MultiIndex.from_frame(f[['order_id','product_id']]).isin(labelkeys).astype('int8')
        if name!='test':
            check(name+'_candidate_label_coverage',int(f.label.sum())==int(tt.truth_count.sum()))
        check(name+'_finite_features',np.isfinite(f[NUMERIC].to_numpy()).all())
        check(name+'_candidate_unique',not f.duplicated(['order_id','product_id']).any())
        check(name+'_all_target_orders_have_candidates',f.order_id.nunique()==len(tt))
        for c in NUMERIC:f[c]=f[c].astype('float32')
        f=f[['user_id','order_id','product_id','group','label']+NUMERIC+CATEGORICAL].sort_values(['group','product_id'])
        f.to_parquet(private/f'{name}.parquet',index=False)
        eda[name]={'users':len(tt),'candidates':len(f),'candidate_positive_rate':float(f.label.mean()) if name!='test' else None,'none_order_rate':float(tt.truth_count.eq(0).mean()) if name!='test' else None,'feature_memory_bytes':int(f.memory_usage(deep=True).sum())}
        log(f'{name}: {len(tt):,} users, {len(f):,} candidates')
        del f;gc.collect()
    write_json(out/'join_leakage_audit.json',audit);write_json(out/'eda_summary.json',eda)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].bar(list(eda['prior_hour_counts']),list(eda['prior_hour_counts'].values()));ax[0,0].set(title='Historical orders by hour',xlabel='Recorded hour',ylabel='Orders')
    ax[0,1].bar(list(eda['prior_gap_counts']),list(eda['prior_gap_counts'].values()));ax[0,1].set(title='Recorded gap (30 = 30 or more days)',xlabel='Days',ylabel='Historical orders')
    t=top.head(10).iloc[::-1];ax[1,0].barh(t.product_name,t.prior_purchases);ax[1,0].set(title='Top products in prior only',xlabel='Historical purchases')
    rates=[eda[k]['candidate_positive_rate'] for k in ['train','tune','hold']]
    ax[1,1].bar(['Train','Tune','Holdout'],rates);ax[1,1].set(title='Positive fraction among historical candidates',ylabel='Fraction',ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'figures'/'eda.png',dpi=140);plt.close(fig)
    (private/'prepared.ok').write_text('complete')

def load_features(private,part):
    f=pd.read_parquet(private/f'{part}.parquet')
    for c in CATEGORICAL:f[c]=f[c].astype('category')
    return f

def train(private,out):
    log('Fitting explanatory logistic baseline on all train candidates')
    f=load_features(private,'train');x=f[NUMERIC].to_numpy(dtype='float32');y=f.label.to_numpy()
    scaler=StandardScaler().fit(x)
    xs=scaler.transform(x).astype('float32')
    logistic=SGDClassifier(loss='log_loss',alpha=.0001,max_iter=30,tol=.0001,random_state=SEED,average=True)
    logistic.fit(xs,y)
    pd.DataFrame({'feature':NUMERIC,'standardized_coefficient':logistic.coef_[0]}).to_csv(out/'logistic_coefficients.csv',index=False)
    log(f'Logistic epochs: {logistic.n_iter_}')
    del xs,x;gc.collect()
    tune=load_features(private,'tune')
    p_log=logistic.predict_proba(scaler.transform(tune[NUMERIC].to_numpy(dtype='float32')))[:,1].astype('float32')
    np.save(private/'tune_logistic.npy',p_log)
    log('Comparing fixed-complexity LightGBM (180 trees, train users only)')
    params=dict(n_estimators=180,learning_rate=.06,num_leaves=31,max_depth=-1,min_child_samples=150,colsample_bytree=1,subsample=1,reg_lambda=5,n_jobs=6,random_state=SEED,verbosity=-1,deterministic=True,force_col_wise=True)
    model=lgb.LGBMClassifier(**params)
    model.fit(f[NUMERIC+CATEGORICAL],y,categorical_feature=CATEGORICAL)
    model.booster_.save_model(str(private/'train_lightgbm.txt'))
    p_lgb=model.predict_proba(tune[NUMERIC+CATEGORICAL])[:,1].astype('float32')
    np.save(private/'tune_lightgbm.npy',p_lgb)
    pd.DataFrame({'feature':NUMERIC+CATEGORICAL,'gain':model.booster_.feature_importance(importance_type='gain')}).sort_values('gain',ascending=False).to_csv(out/'feature_importance.csv',index=False)
    train_meta={'logistic':{'algorithm':'StandardScaler + averaged SGD logistic','alpha':.0001,'max_iter':30,'actual_epochs':int(logistic.n_iter_),'rows':len(f),'class_weight':None},'lightgbm':params,'numeric_features':NUMERIC,'categorical_features':CATEGORICAL}
    del f,y;gc.collect()
    order=pd.read_parquet(private/'tune_orders.parquet');n=len(order)
    g=tune.group.to_numpy();truth=tune.label.to_numpy()
    probs={'frequency':tune.frequency.to_numpy(),'logistic':p_log,'lightgbm':p_lgb}
    rows=[];best={}
    # The independent-product probability of an empty basket is only a tuning heuristic.
    # Test both pure threshold and optional None+products. No holdout feedback is used.
    for name,p in probs.items():
        pn=np.exp(np.bincount(g,weights=np.log1p(-np.clip(p,1e-7,1-1e-7)),minlength=n))
        for threshold in np.round(np.arange(.10,.501,.02),2):
            for nt in [.05,.10,.20,.35,.50,1.1]:
                v=grouped_f1(g,truth,p,n,threshold,pn,nt)
                rows.append({'model':name,'product_threshold':float(threshold),'none_threshold':nt,'mean_order_f1':float(v.mean())})
        b=max([r for r in rows if r['model']==name],key=lambda r:r['mean_order_f1'])
        best[name]=b
        log(f'Tuning {name}: {b}')
    grid=pd.DataFrame(rows);grid.to_csv(out/'threshold_search.csv',index=False)
    winner=max(best,key=lambda k:best[k]['mean_order_f1'])
    chosen=best[winner]
    write_json(out/'selection_locked.json',{'seed':SEED,'tuning_users':n,'selected':chosen,'models':best,'training':train_meta,'holdout_used_for_selection':False})
    fig,ax=plt.subplots(figsize=(8,5))
    for name in probs:
        z=grid[grid.model.eq(name)].groupby('product_threshold').mean_order_f1.max();ax.plot(z.index,z.values,label=name)
    ax.set(xlabel='Product threshold',ylabel='Tuning mean F1 across orders',title='Threshold search on tuning users only');ax.legend();fig.tight_layout();fig.savefig(out/'figures'/'thresholds.png',dpi=140);plt.close(fig)
    del tune,probs,p_log,p_lgb;gc.collect()
    # Selection file is written before this held-out feature/label table is opened.
    log('Locked selection; opening untouched user holdout for final evaluation')
    h=load_features(private,'hold');ho=pd.read_parquet(private/'hold_orders.parquet')
    hp={'frequency':h.frequency.to_numpy(),'logistic':logistic.predict_proba(scaler.transform(h[NUMERIC].to_numpy(dtype='float32')))[:,1].astype('float32'),'lightgbm':model.predict_proba(h[NUMERIC+CATEGORICAL])[:,1].astype('float32')}
    val=[];score_by_model={};g=h.group.to_numpy();truth=h.label.to_numpy();n=len(ho)
    for name,p in hp.items():
        b=best[name];pn=np.exp(np.bincount(g,weights=np.log1p(-np.clip(p,1e-7,1-1e-7)),minlength=n))
        score=grouped_f1(g,truth,p,n,b['product_threshold'],pn,b['none_threshold'])
        score_by_model[name]=score
        val.append({'model':name,'selected_on_tuning':name==winner,'holdout_mean_order_f1':float(score.mean()),'tuning_mean_order_f1':b['mean_order_f1'],'candidate_average_precision':float(average_precision_score(truth,p)),'candidate_log_loss':float(log_loss(truth,np.clip(p,1e-7,1-1e-7))),'product_threshold':b['product_threshold'],'none_threshold':b['none_threshold']})
    pd.DataFrame(val).to_csv(out/'validation_results.csv',index=False)
    scores=score_by_model[winner];rng=np.random.default_rng(SEED+2)
    boot=np.array([rng.choice(scores,len(scores),replace=True).mean() for _ in range(1000)])
    ci=np.quantile(boot,[.025,.975]).tolist()
    allnone=ho.truth_count.eq(0).to_numpy().astype(float)
    lastprob=h.orders_since_last.eq(1).to_numpy().astype(float)
    lastscore=grouped_f1(g,truth,lastprob,n,.5)
    summary={'selected_model':winner,'holdout_orders':n,'mean_order_f1':float(scores.mean()),'bootstrap_95_percent_ci':ci,'bootstrap_replicates':1000,'all_none_baseline':float(allnone.mean()),'repeat_last_basket_baseline':float(lastscore.mean()),'selected_threshold':chosen,'no_reselection_after_holdout':True}
    write_json(out/'validation_summary.json',summary)
    ho[['order_id','user_id','truth_count']].assign(f1=scores).to_parquet(private/'holdout_order_scores.parquet',index=False)
    slices=[]
    for low,high in [(0,0),(1,4),(5,9),(10,1000)]:
        mask=ho.truth_count.between(low,high).to_numpy()
        slices.append({'true_reordered_items':f'{low}-{high}','orders':int(mask.sum()),'mean_f1':float(scores[mask].mean())})
    pd.DataFrame(slices).to_csv(out/'validation_slices.csv',index=False)
    log(f'Final holdout: {summary}')
    del h,hp;gc.collect()
    log('Refitting selected model on all labeled users with frozen settings')
    test=load_features(private,'test');to=pd.read_parquet(private/'test_orders.parquet')
    if winner=='frequency': pt=test.frequency.to_numpy()
    else:
        allf=pd.concat([load_features(private,k) for k in ['train','tune','hold']],ignore_index=True)
        if winner=='lightgbm':
            for c in CATEGORICAL:allf[c]=allf[c].astype('category')
            final=lgb.LGBMClassifier(**params);final.fit(allf[NUMERIC+CATEGORICAL],allf.label,categorical_feature=CATEGORICAL)
            final.booster_.save_model(str(private/'final_lightgbm.txt'))
            pt=final.predict_proba(test[NUMERIC+CATEGORICAL])[:,1].astype('float32')
        else:
            xx=allf[NUMERIC].to_numpy(dtype='float32');ss=StandardScaler().fit(xx)
            final=SGDClassifier(loss='log_loss',alpha=.0001,max_iter=30,tol=.0001,random_state=SEED,average=True)
            final.fit(ss.transform(xx),allf.label);pt=final.predict_proba(ss.transform(test[NUMERIC].to_numpy(dtype='float32')))[:,1]
        del allf;gc.collect()
    g=test.group.to_numpy();n=len(to)
    pn=np.exp(np.bincount(g,weights=np.log1p(-np.clip(pt,1e-7,1-1e-7)),minlength=n))
    kept=test[pt>=chosen['product_threshold']]
    product_lists=kept.groupby('group',sort=False).product_id.agg(list).to_dict()
    result=[]
    for i,oid in enumerate(to.order_id):
        vals=[str(v) for v in product_lists.get(i,[])]
        if not vals or pn[i]>=chosen['none_threshold']:vals.append('None')
        result.append((int(oid),' '.join(vals)))
    submission=pd.DataFrame(result,columns=['order_id','products'])
    assert submission.order_id.is_unique and set(submission.order_id)==set(to.order_id)
    assert len(submission)==75000 and submission.products.ne('').all()
    knownkeys=set(zip(test.order_id,test.product_id))
    for row in submission.itertuples(index=False):
        tokens=row.products.split();assert len(tokens)==len(set(tokens))
        assert all(t=='None' or (row.order_id,int(t)) in knownkeys for t in tokens)
    submission.to_csv(out/'submission.csv',index=False)
    write_json(out/'submission_checks.json',{'pass':True,'rows':len(submission),'columns':list(submission.columns),'all_test_order_ids_exact':True,'all_products_from_user_history':True,'duplicate_tokens':0,'empty_cells':0,'none_only_orders':int(submission.products.eq('None').sum()),'none_with_products_orders':int(submission.products.str.contains(' None').sum()),'sha256':digest(out/'submission.csv'),'official_sample_order_alignment':'test order_id set derived from orders.csv; official sample unavailable without competition re-entry'})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--private',type=Path,required=True);parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--stage',choices=['all','prepare','train'],default='all');args=parser.parse_args()
    args.private.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True);(args.output/'figures').mkdir(exist_ok=True)
    started=time.time()
    if args.stage in ['all','prepare']:prepare(args.data,args.private,args.output)
    if args.stage in ['all','train']:train(args.private,args.output)
    mem=psutil.Process().memory_info()
    write_json(args.output/f'runtime_{args.stage}.json',{'seconds':time.time()-started,'peak_working_set_bytes':getattr(mem,'peak_wset',None),'rss_bytes_at_end':mem.rss,'python':platform.python_version(),'platform':platform.platform(),'seed':SEED})
    log('Completed')

if __name__=='__main__':main()
