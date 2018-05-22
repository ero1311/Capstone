import gdax
from datetime import datetime

public_client = gdax.PublicClient()
#rates=public_client.get_product_historic_rates('BTC-USD',granularity=6
months=[str(i) for i in range(1,13)]
years=[2017]
max_days=[31,28,31,30,31,30,31,31,30,31,30,31]
year_cal=dict(zip(months,max_days))

timestamps=[]
for year in years:
    for month in months:
        if year%4==0 and month=='2':
            max_day=year_cal[month]+2
        else:
            max_day=year_cal[month]+1
        for day in range(1,max_day):
            for hour in hours:
                for minute in mins:
                    timestamps.append(datetime(year,int(month),day,hour,minute,0).isoformat()

count=0
data=[]
for year in years:
    for month in months:
        if year%4==0 and month=='2':
            max_day=year_cal[month]+2
        else:
            max_day=year_cal[month]+1
        for day in range(1,max_day):
            rates=public_client.get_product_historic_rates('BTC-USD',granularity=900,start=datetime(year,int(month),day,0,0,0),end=datetime(year,int(month),day,23,59,59))
            if len(rates)==96:
                count+=1
            data+=rates

import numpy as np
data_np=np.array(data[0]).reshape(1,6)

datetime.fromtimestamp(data_np[0][0]).strftime('%Y-%m-%d %H:%M:%S')

exclude=[]
for i in range(1,len(data)):
    if len(data[i])!=6:
        exclude.append(i)

for i in range(1,len(data)):
    isIn=False
    for ind in exclude:
        if i==ind:
            isIn=True
            break
    if not isIn:
        data_np=np.concatenate((data_np,np.array(data[i]).reshape(1,6)),axis=0)

timestamps=[datetime.fromtimestamp(int(i)).strftime('%Y-%m-%d %H:%M:%S') for i in data_np[:,0]]
import pandas as pd
df=pd.DataFrame(index=timestamps, data=data_np[:,1:],columns=['low', 'high', 'open', 'close', 'volume'])
df.sort_index(inplace=True)
df.to_csv("btc_usd17-18_gran900.csv")

