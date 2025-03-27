import matplotlib.pyplot as plt
import umap
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances

from isomap_train_two_models.synthetic_geometries.data_generation import *

for geometry in geometries.keys():
    data, colors = geometries[geometry]()

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(data)

    isomap = Isomap(n_components=2)
    dist = pairwise_distances(data)
    isomap_embedding = isomap.fit_transform(dist)

    fig, axs = plt.subplots(1, 2, figsize=(8, 5))
    axs[0].scatter(embedding[:, 1], embedding[:, 0], c=colors)
    axs[0].set_title('UMAP')
    axs[1].scatter(isomap_embedding[:, 1], isomap_embedding[:, 0], c=colors)
    axs[1].set_title('ISOMAP')
    plt.suptitle(geometry)
    plt.show()