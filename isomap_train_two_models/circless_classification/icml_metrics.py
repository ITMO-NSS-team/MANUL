import os.path

import pandas as pd

test_acc = []



folder = 'ICML_RESULTS'
for exp in os.listdir(folder):
    df = pd.read_csv(f'{folder}/{exp}/metrics.csv')
    test_acc.append(df['acc_test'].tolist()[0])