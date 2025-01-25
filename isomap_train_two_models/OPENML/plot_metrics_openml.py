import os

import pandas as pd
from matplotlib import pyplot as plt


def binary_classification():
    folder = 'binary_classification/ICML_RESULTS'
    for dataset in os.listdir(folder):
        if 'broken' not in dataset:
            isomap_optim = {'BCE_train': [],
                            'BCE_validation': [],
                            'BCE_test': [],
                            'accuracy_train': [],
                            'accuracy_validation': [],
                            'accuracy_test': []}
            isomap_raw = {'BCE_train': [],
                            'BCE_validation': [],
                            'BCE_test': [],
                            'accuracy_train': [],
                            'accuracy_validation': [],
                            'accuracy_test': []}
            raw_model = {'BCE_train': [],
                            'BCE_validation': [],
                            'BCE_test': [],
                            'accuracy_train': [],
                            'accuracy_validation': [],
                            'accuracy_test': []}

            for exp in os.listdir(f'{folder}/{dataset}'):
                try:
                    isomap_optim_df = pd.read_csv(f'{folder}/{dataset}/{exp}/metrics.csv')
                    for col in isomap_optim_df.columns:
                        isomap_optim[col].append(isomap_optim_df[col].tolist()[0])

                    isomap_raw_df = pd.read_csv(f'{folder}/{dataset}/{exp}/euql_isomap_metrics.csv')
                    for col in isomap_raw_df.columns:
                        isomap_raw[col].append(isomap_raw_df[col].tolist()[0])

                    raw_model_df = pd.read_csv(f'{folder}/{dataset}/{exp}/raw_model_metrics.csv')
                    for col in raw_model_df.columns:
                        raw_model[col].append(raw_model_df[col].tolist()[0])
                except Exception:
                    pass

            df_accuracy_train = pd.DataFrame()
            df_accuracy_train['raw_model'] = raw_model['accuracy_train']
            df_accuracy_train['isomap_raw'] = isomap_raw['accuracy_train']
            df_accuracy_train['isomap_optim'] = isomap_optim['accuracy_train']
            df_accuracy_train.to_csv(f'{folder}/{dataset}/accuracy_train.csv', index=False)

            df_accuracy_test = pd.DataFrame()
            df_accuracy_test['raw_model'] = raw_model['accuracy_test']
            df_accuracy_test['isomap_raw'] = isomap_raw['accuracy_test']
            df_accuracy_test['isomap_optim'] = isomap_optim['accuracy_test']
            df_accuracy_test.to_csv(f'{folder}/{dataset}/accuracy_test.csv', index=False)

            df_accuracy_validation = pd.DataFrame()
            df_accuracy_validation['raw_model'] = raw_model['accuracy_validation']
            df_accuracy_validation['isomap_raw'] = isomap_raw['accuracy_validation']
            df_accuracy_validation['isomap_optim'] = isomap_optim['accuracy_validation']
            df_accuracy_validation.to_csv(f'{folder}/{dataset}/accuracy_validation.csv', index=False)

            # _________ BCE save______________
            df_bce_train = pd.DataFrame()
            df_bce_train['raw_model'] = raw_model['BCE_train']
            df_bce_train['isomap_raw'] = isomap_raw['BCE_train']
            df_bce_train['isomap_optim'] = isomap_optim['BCE_train']
            df_bce_train.to_csv(f'{folder}/{dataset}/BCE_train.csv', index=False)

            df_bce_test = pd.DataFrame()
            df_bce_test['raw_model'] = raw_model['BCE_test']
            df_bce_test['isomap_raw'] = isomap_raw['BCE_test']
            df_bce_test['isomap_optim'] = isomap_optim['BCE_test']
            df_bce_test.to_csv(f'{folder}/{dataset}/BCE_test.csv', index=False)

            df_bce_validation = pd.DataFrame()
            df_bce_validation['raw_model'] = raw_model['BCE_validation']
            df_bce_validation['isomap_raw'] = isomap_raw['BCE_validation']
            df_bce_validation['isomap_optim'] = isomap_optim['BCE_validation']
            df_bce_validation.to_csv(f'{folder}/{dataset}/BCE_validation.csv', index=False)


            fig, ax = plt.subplots()
            plt.ylabel('Accuracy')
            ax.boxplot([raw_model['accuracy_train'],
                        isomap_raw['accuracy_train'],
                        isomap_optim['accuracy_train']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Train')
            plt.savefig(f'{folder}/{dataset}/train_accuracy.png')
            plt.close()

            fig, ax = plt.subplots()
            plt.ylabel('Accuracy')
            ax.boxplot([raw_model['accuracy_test'],
                        isomap_raw['accuracy_test'],
                        isomap_optim['accuracy_test']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Test')
            plt.savefig(f'{folder}/{dataset}/test_accuracy.png')
            plt.close()

            fig, ax = plt.subplots()
            plt.ylabel('Accuracy')
            ax.boxplot([raw_model['accuracy_validation'],
                        isomap_raw['accuracy_validation'],
                        isomap_optim['accuracy_validation']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Validation')
            plt.savefig(f'{folder}/{dataset}/validation_accuracy.png')
            plt.close()

            fig, ax = plt.subplots()
            plt.ylabel('BCE')
            ax.boxplot([raw_model['BCE_validation'],
                        isomap_raw['BCE_validation'],
                        isomap_optim['BCE_validation']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Validation')
            plt.savefig(f'{folder}/{dataset}/validation_bce.png')
            plt.close()

            fig, ax = plt.subplots()
            plt.ylabel('BCE')
            ax.boxplot([raw_model['BCE_train'],
                        isomap_raw['BCE_train'],
                        isomap_optim['BCE_train']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Train')
            plt.savefig(f'{folder}/{dataset}/train_bce.png')
            plt.close()

            fig, ax = plt.subplots()
            plt.ylabel('BCE')
            ax.boxplot([raw_model['BCE_test'],
                        isomap_raw['BCE_test'],
                        isomap_optim['BCE_test']])
            ax.set_xticklabels(['Raw model', 'Model+ISOMAP', 'Optimized ISOMAP'])
            plt.title(f'{dataset.split(".")[0]} - Test')
            plt.savefig(f'{folder}/{dataset}/test_bce.png')
            plt.close()



binary_classification()