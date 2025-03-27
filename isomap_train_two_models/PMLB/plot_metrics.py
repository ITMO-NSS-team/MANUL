import os

import pandas as pd
from matplotlib import pyplot as plt


def regression():
    folder = 'regression'
    for dataset in os.listdir(folder):
        isomap_optim = {'train_mae': [],
                        'val_mae': [],
                        'test_mae': [],
                        'train_mse': [],
                        'val_mse': [],
                        'test_mse': []}
        isomap_raw = {'train_mae': [],
                      'val_mae': [],
                      'test_mae': [],
                      'train_mse': [],
                      'val_mse': [],
                      'test_mse': []}
        raw_model = {'train_mae': [],
                     'val_mae': [],
                     'test_mae': [],
                     'train_mse': [],
                     'val_mse': [],
                     'test_mse': [],
                     }
        linear_regression = {'train_mae': [],
                             'val_mae': [],
                             'test_mae': [],
                             'train_mse': [],
                             'val_mse': [],
                             'test_mse': []}

        for r in range(5):
            try:
                df = pd.read_csv(f'{folder}/{dataset}/isomap_nn/{r}_isomap_optimized_metrics.csv')
                for col in df.columns:
                    isomap_optim[col].append(df[col].tolist()[0])

                df = pd.read_csv(f'{folder}/{dataset}/isomap_nn/{r}_isomap_raw_metrics.csv')
                for col in df.columns:
                    isomap_raw[col].append(df[col].tolist()[0])

                df = pd.read_csv(f'{folder}/{dataset}/linear_nn/{r}_Linear_NN_metrics.csv')
                for col in df.columns:
                    raw_model[col].append(df[col].tolist()[0])

                df = pd.read_csv(f'{folder}/{dataset}/linear_regression/{r}_Linear_reg_metrics.csv')
                for col in df.columns:
                    linear_regression[col].append(df[col].tolist()[0])

                df_train = pd.DataFrame()
                df_train['linear_regression'] = linear_regression['train_mae']
                df_train['raw_model'] = raw_model['train_mae']
                df_train['isomap_raw'] = isomap_raw['train_mae']
                df_train['isomap_optim'] = isomap_optim['train_mae']
                df_train.to_csv(f'{folder}/{dataset}/train_mae.csv', index=False)

                df_test = pd.DataFrame()
                df_test['linear_regression'] = linear_regression['test_mae']
                df_test['raw_model'] = raw_model['test_mae']
                df_test['isomap_raw'] = isomap_raw['test_mae']
                df_test['isomap_optim'] = isomap_optim['test_mae']
                df_test.to_csv(f'{folder}/{dataset}/test_mae.csv', index=False)

                try:
                    df_val = pd.DataFrame()
                    df_val['linear_regression'] = linear_regression['val_mae']
                    df_val['raw_model'] = raw_model['val_mae']
                    df_val['isomap_raw'] = isomap_raw['val_mae']
                    df_val['isomap_optim'] = isomap_optim['val_mae']
                    df_val.to_csv(f'{folder}/{dataset}/val_mae.csv', index=False)
                except Exception as e:
                    print(e)
                    pass

                fig, ax = plt.subplots()
                plt.ylabel('MAE')
                ax.boxplot([linear_regression['train_mae'],
                            raw_model['train_mae'],
                            isomap_raw['train_mae'],
                            isomap_optim['train_mae']])
                ax.set_xticklabels(['Linear regression', 'Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
                plt.title(f'{dataset.split(".")[0]} - Train')
                plt.savefig(f'{folder}/{dataset}/train_mae.png')
                plt.close()

                fig, ax = plt.subplots()
                plt.ylabel('MAE')
                ax.boxplot([linear_regression['test_mae'],
                            raw_model['test_mae'],
                            isomap_raw['test_mae'],
                            isomap_optim['test_mae']])
                ax.set_xticklabels(['Linear regression', 'Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
                plt.title(f'{dataset.split(".")[0]} - Test')
                plt.savefig(f'{folder}/{dataset}/test_mae.png')
                plt.close()
            except Exception as e:
                print(e)
                pass

regression()