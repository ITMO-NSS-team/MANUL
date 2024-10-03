import os
import sys

root_dir = '/'.join(os.getcwd().split("/")[:-1])
sys.path.append(root_dir)

from torchvision import datasets
import numpy as np
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


dataset = datasets.MNIST('examples/data', train=True, download=False)
train_data = dataset.train_data.numpy()
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = dataset.train_labels.numpy()

test_data = dataset.test_data.numpy()
test_data = test_data.reshape((test_data.shape[0], test_data.shape[1] * test_data.shape[2]))
test_labels = dataset.test_labels.numpy()

train_labels = train_labels[:15000]
train_data = train_data[:15000, :]

test_labels = test_labels[:3000]
test_data = test_data[:3000, :]

cache_folder = 'cache_siam_v2'



# SIAM MODEL
siam_matrix_connect = np.load('examples/data/siamese_MNIST_train_set.npy')
siam_matrix_connect = siam_matrix_connect[:15000, :15000]

# upper diagonal
print('calc 2nd part siam')
mask_matrix = np.full(siam_matrix_connect.shape, 1)
mask_matrix = np.tril(mask_matrix, 1)
siam_matrix_connect2 = mask_matrix * siam_matrix_connect
siam_matrix_connect2 = np.maximum(siam_matrix_connect2, siam_matrix_connect2.transpose())
siam_matrix_connect2 = siam_matrix_connect2/np.max(siam_matrix_connect2)

siam_adj_matrix2 = np.ones(siam_matrix_connect2.shape)
np.fill_diagonal(siam_adj_matrix2, 0)
siam_adj_matrix2[siam_matrix_connect2 == 0] = 0

siam_individ = DataStructureGraph(cache_folder=cache_folder)
siam_individ.basis = np.arange(siam_adj_matrix2.shape[0])
siam_individ.matrix_connect = siam_matrix_connect2[siam_individ.basis][:, siam_individ.basis]
siam_individ.adjacency_matrix = siam_adj_matrix2[siam_individ.basis][:, siam_individ.basis]

siam_individ.source_data = train_data

siam_model = ModelNN(train_data, train_labels,
                     num_epochs=50,
                     batch_size=300,
                     problem='multiclass',
                     cache_folder=cache_folder,
                     model_name='siam2'
                     )
siam_model.train(siam_individ)

with open(f'{cache_folder}/siam2_metrics.txt', 'a') as file:
    file.write(f'f1 score for siam2):\n')
    file.write(f'Train {siam_model.get_metric_on_train()}\n')
    file.write(f'Test {siam_model.get_metric_on_test(test_data, test_labels)}\n')

print(f'f1 score for siam2:')
print('Train')
print(siam_model.get_metric_on_train())
print('Test')
print(siam_model.get_metric_on_test(test_data, test_labels))