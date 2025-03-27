import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from torch import float32

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.synthetic_geometries.data_generation import geometries

data, labels = geometries['sphere']()

X_train, X_test, y_train, y_test = train_test_split(data, labels)

embedding = Isomap(n_components=3)

train_dist = pairwise_distances(X_train, X_train)
max_val = np.max(train_dist)
train_dist = train_dist/max_val
X_train_transformed = embedding.fit_transform(train_dist)
X_train_transformed_2 = embedding.transform(train_dist)

test_dist = pairwise_distances(X_test, X_train)
test_dist = test_dist/max_val
X_test_transformed = embedding.transform(test_dist)

print('TRAIN')
print(f'var = {np.var(X_train_transformed)}')
print(f'std = {np.std(X_train_transformed)}')
print('________________')
print('TEST')
print(f'var = {np.var(X_test_transformed)}')
print(f'std = {np.std(X_test_transformed)}')


isomap = IsomapNN(torch.tensor(train_dist, dtype=float32).to('cuda'), n_components=3)
X_train_transformed = isomap()

X_test_transformed = isomap.transform(torch.tensor(test_dist, dtype=float32).to('cuda'))

print('____________Torch isomap__________')
print('TRAIN')
print(f'var = {np.var(X_train_transformed.cpu().detach().numpy())}')
print(f'std = {np.std(X_train_transformed.cpu().detach().numpy())}')
print('________________')
print('TEST')
print(f'var = {np.var(X_test_transformed.cpu().detach().numpy())}')
print(f'std = {np.std(X_test_transformed.cpu().detach().numpy())}')