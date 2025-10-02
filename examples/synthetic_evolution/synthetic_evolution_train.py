from datetime import datetime

import torch
from torch import float32

from evolution.Evolution import Evolution
from examples.data.synthetic_data_generation import geometries


device = 'cuda'
for geometry in geometries.keys():
    data, labels = geometries[geometry][0]()
    latent_len = geometries[geometry][1]
    train_features = torch.tensor(data, dtype=float32).to(device)
    train_target = torch.tensor(labels, dtype=float32).to(device)

    evolution = Evolution(train_feature=train_features,
                            train_target=train_target,
                            latent_len=latent_len,
                            logs_folder=f'{geometry}_{datetime.now().strftime("%d%m%Y-%H.%M")}',)