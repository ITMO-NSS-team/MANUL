import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from isomap_train_two_models.synthetic_geometries.data_generation import geometries


def create_df_per_run():
    path = 'results_(3k_1var)'
    for geometry in geometries.keys():
        lr_folder = f'{path}/{geometry}/linear_regression'
        isomap_folder = f'{path}/{geometry}/isomap'
        models = {'Linear_reg': lr_folder, 'isomap_raw': isomap_folder, 'isomap_optimized': isomap_folder}

        for m in models.keys():
            folder = models[m]
            full_df = pd.DataFrame()
            full_df['metric'] = ['train_mae', 'train_mse', 'test_mae', 'test_mse']
            for i in range(5):
                df = pd.read_csv(f'{folder}/{i}_{m}_metrics.csv')
                full_df[i] = df.iloc[0].tolist()
            full_df = full_df.T
            full_df.columns = full_df.iloc[0]
            full_df = full_df.drop(full_df.index[0])
            full_df.to_csv(f'{folder}/{m}_metrics.csv', index=False)


def create_df_all_models():
    path = 'results_(3k_1var)'
    for geometry in geometries.keys():
        full_geom_df = pd.DataFrame()

        lr_folder = f'{path}/{geometry}/linear_regression'
        isomap_folder = f'{path}/{geometry}/isomap'

        lr_ds = pd.read_csv(f'{lr_folder}/Linear_reg_metrics.csv')
        isomap_raw_ds = pd.read_csv(f'{isomap_folder}/isomap_raw_metrics.csv')
        isomap_opt_ds = pd.read_csv(f'{isomap_folder}/isomap_optimized_metrics.csv')

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].set_ylabel('MAE')
        axs[1].set_ylabel('MAE')
        axs[0].boxplot([lr_ds['train_mae'].values,
                                isomap_raw_ds['train_mae'].values,
                                isomap_opt_ds['train_mae'].values, ],
                               patch_artist=True,
                               tick_labels=['Linear\nregression',
                                            'ISOMAP\nraw',
                                            'ISOMAP\noptimized'])
        axs[1].boxplot([lr_ds['test_mae'].values,
                        isomap_raw_ds['test_mae'].values,
                        isomap_opt_ds['test_mae'].values, ],
                       patch_artist=True,
                       tick_labels=['Linear\nregression',
                                    'ISOMAP\nraw',
                                    'ISOMAP\noptimized'])
        axs[0].set_title('Train')
        axs[1].set_title('Test')
        plt.suptitle(f'{geometry}')
        plt.tight_layout()
        plt.savefig(f'{path}/{geometry}_mae.png')
        plt.show()

        lr_ds = lr_ds.mean()
        isomap_raw_ds = isomap_raw_ds.mean()
        isomap_opt_ds = isomap_opt_ds.mean()

        full_geom_df['metrics'] = lr_ds.index
        full_geom_df['linear_regression'] = lr_ds.values
        full_geom_df['isomap_raw'] = isomap_raw_ds.values
        full_geom_df['isomap_optimized'] = isomap_opt_ds.values

        full_geom_df.to_csv(f'{path}/{geometry}_metrics.csv', index=False)


#create_df_per_run()
create_df_all_models()
