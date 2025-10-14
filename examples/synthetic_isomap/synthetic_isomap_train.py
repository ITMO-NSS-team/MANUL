from datetime import datetime

import torch
from torch import float32

from examples.data.synthetic_data_generation import geometries
from Adam.GradientIsomap import GradientIsomap

device = 'cuda'
#for geometry in geometries.keys():
for geometry in ['swiss_roll']:
    data, labels = geometries[geometry][0]()
    latent_len = geometries[geometry][1]
    train_features = torch.tensor(data, dtype=float32).to(device)
    train_target = torch.tensor(labels, dtype=float32).to(device)
    isomap = GradientIsomap(train_feature=train_features,
                            train_target=train_target,
                            latent_len=latent_len,
                            checkpoint_each=100,
                            logs_folder=f'test/{geometry}_{datetime.now().strftime("%d%m%Y-%H.%M")}',
                            plot_convergence=False,
                            epochs=500)
    isomap.train()
    isomap.visualize_trained()
    matrix = isomap.best_distances_matrix
    proj_features = isomap.best_isomap_model()
    proj_features
