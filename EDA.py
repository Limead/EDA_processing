import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import shutil
from tqdm import tqdm
from Utils.logger import Logger
from Uitls.utils import *
## 구동코드 : python3 EDA.py -data 데이터명.csv -target Target

# 수치형 시각화
def num_visualization(cols, data, y, eda_path):
    
    boxplot = plt.figure()
    sns.boxplot(data[cols].dropna())
    plt.title(f'Boxplot - {cols}')
    boxplot.savefig(f'{eda_path}/boxplot_{cols}.png')
    plt.close()
    distplot = plt.figure()
    sns.distplot(data[cols].dropna())
    plt.title(f'Distplot - {cols}')
    distplot.savefig(f'{eda_path}/distplot_{cols}.png')
    plt.close()
    scatter= plt.figure()
    sns.scatterplot(data[cols].dropna())
    plt.title(f'Scatterplot- {cols}')
    scatter.savefig(f'{eda_path}/scatterplot_{cols}.png')
    plt.close()
    
# 범주형 시각화
def obj_visualization(cols, data, target_name, eda_path):
    data.groupby([target_name])[cols].hist()
    plt.title(f'y - {cols} histogram')
    plt.xlabel(cols)
    plt.tight_layout()
    plt.savefig(f'{eda_path}/y_{cols}_hist.png')
    plt.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('parser')
    parser.add_argument('-data', dest = 'data', required=True)
    parser.add_argument('-eda_path', dest='eda_path', default='/EDA_results')
    parser.add_argument('-target', dest='target', required=True)
    args = parser.parse_args()
    data_path = args.data
    eda_path = args.eda_path
    target_name = args.target
    log = Logger('EDA_log')
    
    try:
        # data path logging
        log.info(data_path)
        log.info(f'Target : {target_name}')
    
        if os.path.exists(eda_path):
            shutil.rmtree(eda_path)
        if not os.path.exists(eda_path):
            os.makedirs(eda_path)
        ## EDA ##
        data = pd.read_csv(data_path)
        target = data[f'{target_name}']
        # 통계량 산출
        Describe_df = data.describe()
        Null_counts = pd.isnull(data).sum()
        df = Describe_df.transpose()
        df['NA_counts'] = Null_counts
        Describe_df = df.transpose()
        Describe_df.to_csv(f'{eda_path}/Describe_df.csv')
        data_corr().to_csv(f'{eda_path}/corr_df.csv')
    
        # 모든 행이 동일한 값을 갖는 컬럼 제거
        del_cols = (data.nunique() == 1).loc[(data.nunique()==1) == True].index.tolist()
        data.drop(columns = del_cols, inplace=True)
    
        # 변수별 EDA
        for cols in tqdm(data.drop(columns=target_name)):
            log.info(f'{cols} : {data[cols].dtype}')
            # 수치형 변수
            if (data[cols].dtype == 'int64' and cols[-2:] not in ['CD','ID']) or ( data[cols].dtype == 'float64' and cols[-2:] not in ['CD','ID']):
                if pd.isnull(data[cols]).sum()/len(data[cols]) > 0.9: # 수치형 중 결측이 90% 이상인 경우 제외
                    continue
                
                num_visualization(cols, data, target, eda_path)
            
            # 범주형 변수
            elif data[cols].dtype == 'object' or cols[-2:] == 'CD':
                if data[cols].nunique() > 100 or len(str(data[cols][0])) > 10: # ID와 같이 범주가 너무 많거나 문장 형태(자연어) 인 컬럼 제외
                    continue
                
                obj_visualization(cols, data, target_name, eda_path)
            else:
                continue
    except Exception as e:
        log.error(e)
    finally:
        log.info('finish')
