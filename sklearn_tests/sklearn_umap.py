import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from sklearn.manifold import Isomap

penguins = pd.read_csv(
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
penguins = penguins.dropna()
penguins.species.value_counts()

reducer = umap.UMAP()

penguin_data = penguins[
    [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values
scaled_penguin_data = StandardScaler().fit_transform(penguin_data)

embedding = reducer.fit_transform(scaled_penguin_data)

plt.scatter(embedding[:, 1], embedding[:, 0],
            c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
plt.title('UMAP sklearn transform')
plt.show()


pca_embedding = PCA(n_components=2).fit_transform(penguin_data)
plt.scatter(pca_embedding[:, 1], pca_embedding[:, 0],
            c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
plt.title('PCA sklearn transform')
plt.show()

embedding = Isomap(n_components=2)
dist = pairwise_distances(penguin_data)
isomap_embedding = embedding.fit_transform(dist)
plt.scatter(isomap_embedding[:, 1], isomap_embedding[:, 0],
            c=[sns.color_palette()[x] for x in penguins.species.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})])
plt.title('ISOMAP sklearn transform')
plt.show()