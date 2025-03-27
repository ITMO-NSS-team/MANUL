import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, float32
from torchcnnbuilder.preprocess.time_series import multi_output_tensor

from isomap_pytorch.Isomap import IsomapNN

device = 'cuda'

df = pd.read_csv('50_50_kara.csv')
df['dates'] = pd.to_datetime(df['dates'])
train_df = df[df['dates'].dt.year < 2020]
test_df = df[df['dates'].dt.year >= 2020]


train_ts = np.array(train_df['ice_conc'].values)[::7]
train_dates = np.array(train_df['dates'].values)[::7]

train_dataset = multi_output_tensor(data=train_ts,
                                    pre_history_len=104,
                                    forecast_len=52)

train_dates_inds = np.arange(len(train_dates))
train_dates_dataset = multi_output_tensor(data=train_dates_inds,
                                    pre_history_len=104,
                                    forecast_len=52)

train_features = train_dataset.tensors[0]
train_target = train_dataset.tensors[1]
dist = torch.cdist(torch.tensor(train_features), torch.tensor(train_features))

exp_path = f'ts_ice_forecasting/isomap'
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

# ________________TRAIN ISOMAP ______________________________
latent_len = 15
isomap_model = IsomapNN(dist, n_components=latent_len)
isomap_model.to(device)

isomap_epochs = 2000
task_epochs = 150
isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.001)
isomap_criterion = nn.L1Loss()
losses = []
best_loss = np.inf
best_isomap_model = None

# ISOMAP TRAIN LOOP
for epoch in range(isomap_epochs):
    reproj_features = isomap_model().to(float32)

    with torch.no_grad():
        features = isomap_model.transform(dist)

    task_model = nn.Sequential(nn.Linear(latent_len, 562, dtype=float32),
                               #nn.ReLU(),
                               nn.Linear(562, 256, dtype=float32),
                               nn.Linear(256, 64, dtype=float32),
                               nn.Linear(64, 1, dtype=float32)
                               ).to(device)
    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
    task_criterion = nn.L1Loss()

    task_losses = []
    for ep in range(task_epochs):
        task_optim.zero_grad()
        out = task_model(features)
        task_loss = task_criterion(out, train_target.reshape_as(out))
        # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
        task_losses.append(task_loss.item())
        task_loss.backward()
        task_optim.step()

    output = task_model(reproj_features)
    isomap_loss = isomap_criterion(output.to(torch.float32),
                                   reduced_train_target.reshape_as(output).to(torch.float32))
    losses.append(isomap_loss.item())

    if losses[-1] < best_loss:
        best_loss = losses[-1]
        best_isomap_model = isomap_model


    isomap_loss.backward()
    isomap_optim.step()
    print(f'Epoch {epoch}/{isomap_epochs},  loss={losses[-1]}')
