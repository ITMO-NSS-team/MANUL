import os
import sys

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

from sklearn.datasets import load_digits
from torchvision import datasets
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.metrics import pairwise_distances
import numpy as np
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN

dataset = datasets.MNIST('examples/data', train=True, download=False)
train_data = dataset.train_data.numpy()
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = dataset.train_labels.numpy()

train_labels = train_labels[:15000]
train_data = train_data[:15000, :]


test_data = dataset.test_data.numpy()
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
test_labels = dataset.test_labels.numpy()

train_labels = train_labels[:15000]
train_data = train_data[:15000, :]

test_labels = test_labels[:3000]
test_data = test_data[:3000, :]


outfeatures_num = 10
# 2,5,10


embedding = LocallyLinearEmbedding(n_components=outfeatures_num)
print('process embedding train')
X_transformed = embedding.fit_transform(train_data)
print('process embedding test')
test_features_trans = embedding.fit_transform(test_data)
print('calc distances')
embed_matrix_connect = pairwise_distances(X_transformed)

cache_folder = f'cache_embedd_f_num({outfeatures_num})_v2'

print('calc matrices')

embed_adj_matrix = np.ones(embed_matrix_connect.shape)
np.fill_diagonal(embed_adj_matrix, 0)
embed_adj_matrix[embed_matrix_connect==0] = 0

embed_individ = DataStructureGraph(cache_folder=cache_folder)
embed_individ.basis = np.arange(embed_matrix_connect.shape[0])
embed_individ.matrix_connect = embed_matrix_connect
embed_individ.adjacency_matrix = embed_adj_matrix
embed_individ.source_data = X_transformed

embed_model = ModelNN(X_transformed, train_labels,
                     num_epochs=50,
                     batch_size=300,
                     problem='multiclass',
                     cache_folder=cache_folder,
                     model_name=f'ebmed_{outfeatures_num}'
                     )
embed_model.train(embed_individ, adaptive_lambda=False, plot_convergence=True)

print(f'f1 score for embedding features (num={outfeatures_num}):')
print('Train')
print(embed_model.get_metric_on_train())
print('Test')
print(embed_model.get_metric_on_test(test_features_trans, test_labels))

with open('embedding_metrics.txt', 'a') as file:
    file.write(f'f1 score for embedding features (num={outfeatures_num}):\n')
    file.write(f'Train {embed_model.get_metric_on_train()}\n')
    file.write(f'Test {embed_model.get_metric_on_test(test_features_trans, test_labels)}\n')