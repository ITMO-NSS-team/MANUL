import os

import pandas as pd

path = 'C:/Users/Julia/Documents/NSS_lab/документы/2024 NIPS/dirty_code/openml_paper_statement/regression/results/combined'

for folder in os.listdir(path):
    file = '_'.join(folder.split('_')[1:])+'.csv'
    df = pd.read_csv(f'{path}/{folder}/{file}')
    target_name = df.columns[-1]
    print(f'{file} - {target_name}')