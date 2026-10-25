"""K-Means segmentation demo on synthetic customer records."""
import argparse,json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
FIELDS=['customer_id','age','engagement_score','annual_spend_pkr']
FEATURES=['age','engagement_score','annual_spend_pkr']
def analyze(path,out,k=3):
    df=pd.read_csv(path)
    missing=set(FIELDS)-set(df.columns)
    if missing:raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty or df[FIELDS].isna().any().any():raise ValueError('Empty or missing data')
    if df.customer_id.duplicated().any():raise ValueError('Duplicate customer IDs')
    for col in FEATURES:df[col]=pd.to_numeric(df[col],errors='raise')
    if not df.age.between(18,120).all() or not df.engagement_score.between(0,100).all() or not (df.annual_spend_pkr>=0).all():raise ValueError('Invalid feature ranges')
    if not 2<=k<=len(df):raise ValueError('k must be between 2 and number of records')
    scaled=StandardScaler().fit_transform(df[FEATURES])
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    df['cluster']=model.fit_predict(scaled)
    out.mkdir(parents=True,exist_ok=True)
    profile=df.groupby('cluster',as_index=False).agg(customers=('customer_id','count'),average_age=('age','mean'),average_engagement=('engagement_score','mean'),average_spend_pkr=('annual_spend_pkr','mean')).round(2)
    profile.to_csv(out/'cluster_profile.csv',index=False);df.to_csv(out/'customers_with_clusters.csv',index=False)
    summary={'customers':int(len(df)),'clusters':int(k),'cluster_sizes':{str(int(row.cluster)):int(row.customers) for row in profile.itertuples()},'inertia':round(float(model.inertia_),3)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8,5));sc=ax.scatter(df.engagement_score,df.annual_spend_pkr,c=df.cluster,cmap='viridis',alpha=.7,s=25);ax.set(title='Customer Segments (Synthetic Data)',xlabel='Engagement score (0–100)',ylabel='Annual spend (PKR)');fig.colorbar(sc,ax=ax,label='Cluster ID');fig.tight_layout();fig.savefig(out/'customer_segments.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));ax.bar(profile.cluster.astype(str),profile.customers);ax.set(title='Customers per Cluster',xlabel='Cluster ID',ylabel='Customers');fig.tight_layout();fig.savefig(out/'cluster_sizes.png',dpi=160);plt.close(fig)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/customers.csv'));p.add_argument('--out',type=Path,default=Path('outputs'));p.add_argument('--clusters',type=int,default=3);a=p.parse_args();print(json.dumps(analyze(a.data,a.out,a.clusters),indent=2))
