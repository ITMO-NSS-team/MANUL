import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps
from sklearn.model_selection import train_test_split

from isomap_train_two_models.synthetic_geometries.data_generation import geometries
import plotly.express as px

device = 'cuda'

df = pd.read_csv('50_50_kara.csv')
df['dates'] = pd.to_datetime(df['dates'])
train_df = df[df['dates'].dt.year < 2020]
val_df = df[df['dates'].dt.year >= 2020]

train_ts = np.array(train_df['ice_conc'].values)[::7]
train_dates = np.array(train_df['dates'].values)[::7]

train_dataset = multi_output_tensor(data=train_ts,
                                    pre_history_len=104,
                                    forecast_len=52)
train_dates_inds = np.arange(len(train_dates))
train_dates_dataset = multi_output_tensor(data=train_dates_inds,
                                          pre_history_len=104,
                                          forecast_len=52)
train_features = train_dataset.tensors[0].to(device)
train_target = train_dataset.tensors[1].to(device)

latent_len = train_features.size(1)
out_len = train_target.size(1)
task_model = nn.Sequential(nn.Linear(latent_len, 562, dtype=float32),
                           nn.Linear(562, 256, dtype=float32),
                           nn.Linear(256, 64, dtype=float32),
                           nn.Linear(64, out_len, dtype=float32)
                           ).to(device)
task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
task_criterion = nn.L1Loss()
task_epochs = 2000
losses = []
for ep in range(task_epochs):
    task_optim.zero_grad()
    out = task_model(train_features)
    task_loss = task_criterion(out.reshape_as(train_target), train_target)
    print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
    losses.append(task_loss.item())
    task_loss.backward()
    task_optim.step()

plt.plot(np.arange(task_epochs), losses)
plt.show()

prediction = []
prediction_dates = []

plt.rcParams['figure.figsize'] = (10, 4)
for year in [2020, 2021, 2022, 2023]:
    target_f = df[df['dates'].dt.year == year][::7][-52:]
    features_f = df[(df['dates'].dt.year >= year - 2) & (df['dates'].dt.year < year)][::7][-104:]

    features = features_f['ice_conc'].values
    output = task_model(torch.tensor(features, dtype=float32).to(device)).cpu().detach().numpy()
    output[output > 1] = 1
    output[output < 0] = 0

    prediction.extend(output.tolist())
    prediction_dates.extend(target_f['dates'].values)

    plt.axvline(datetime(year, 1, 1), c='black', linestyle='dashed')

prehist = df[(df['dates'].dt.year > 2018) & (df['dates'].dt.year <2024)]

target = df[df['dates'].isin(prediction_dates)]['ice_conc'].values

mae = np.mean(abs(target - prediction))/len(target)
mse = np.mean((target - prediction)**2)/len(target)

plt.plot(prehist['dates'], prehist['ice_conc'], label='Real', c='green')
plt.plot(prediction_dates, prediction, label='Predicted', c='red')
plt.legend()
plt.title(f'Ice concentration prediction in point\n MAE={np.round(mae, 5)}, MSE={np.round(mse, 5)}')
plt.tight_layout()
plt.show()
