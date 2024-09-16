from torchvision import datasets
import numpy as np
from evolution.IndividStructures import DataStructureGraph
from regularizator.ModuleNN import ModelNN


dataset = datasets.MNIST('data', train=True, download=False)
train_data = dataset.train_data.numpy()
train_data = train_data.reshape((train_data.shape[0], train_data.shape[1] * train_data.shape[2]))
train_labels = dataset.train_labels.numpy()

train_labels = train_labels[:30000]
train_data = train_data[:30000, :]


cach_folder = 'cach_siam'

base_individ = DataStructureGraph(data=train_data,
                                  cach_folder='cach',
                                  n_neighbors=20,
                                  graph_file='base_graph.pkl')

# SIAM MODEL
siam_matrix_connect = np.load('data/siamese_MNIST_train_set.npy')
siam_matrix_connect = siam_matrix_connect[:30000, :30000]

# lower diagonal
print('calc 1st part siam')
mask_matrix = np.full(siam_matrix_connect.shape, 1)
mask_matrix = np.tril(mask_matrix, -1)
siam_matrix_connect1 = mask_matrix * siam_matrix_connect
siam_matrix_connect1 = np.maximum(siam_matrix_connect1, siam_matrix_connect1.transpose())
siam_matrix_connect1 = siam_matrix_connect1/np.max(siam_matrix_connect1)

siam_adj_matrix1 = np.ones(siam_matrix_connect1.shape)
np.fill_diagonal(siam_adj_matrix1, 0)
siam_adj_matrix1[siam_matrix_connect1==0] = 0

siam_individ = DataStructureGraph(cach_folder=cach_folder)
siam_individ.basis = base_individ.basis
siam_individ.matrix_connect = siam_matrix_connect1[siam_individ.basis][:, siam_individ.basis]
siam_individ.adjacency_matrix = siam_adj_matrix1[siam_individ.basis][:, siam_individ.basis]

siam_individ.source_data = base_individ.source_data
siam_individ.show_2d(train_labels, save_path=f'{cach_folder}/siam_graph1.png')
siam_individ.show_3d(train_labels, title='Siamese network')

siam_model = ModelNN(train_data, train_labels,
                     num_epochs=200,
                     batch_size=300,
                     problem='multiclass',
                     cach_folder=cach_folder,
                     model_name='siam1'
                     )
siam_model.train(siam_individ)

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

siam_individ = DataStructureGraph(cach_folder=cach_folder)
siam_individ.basis = base_individ.basis
siam_individ.matrix_connect = siam_matrix_connect2[siam_individ.basis][:, siam_individ.basis]
siam_individ.adjacency_matrix = siam_adj_matrix2[siam_individ.basis][:, siam_individ.basis]

siam_individ.source_data = base_individ.source_data
siam_individ.show_2d(train_labels, save_path=f'{cach_folder}/siam_graph2.png')
siam_individ.show_3d(train_labels, title='Siamese network')

siam_model = ModelNN(train_data, train_labels,
                     num_epochs=200,
                     batch_size=300,
                     problem='multiclass',
                     cach_folder=cach_folder,
                     model_name='siam2'
                     )
siam_model.train(siam_individ)