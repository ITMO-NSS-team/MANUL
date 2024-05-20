import matplotlib.pyplot as plt


import numpy as np
from sklearn.metrics import euclidean_distances

from evolution.IndividStructures import DataStructureGraph

base_individ = DataStructureGraph(data=None,
                                              cash_folder='C:/Users/Julia/Documents/NSS_lab/fastnet/examples/info_log/mammonth_fix',
                                  graph_file='base_graph.pkl')

final_individ = DataStructureGraph(data=None,
                                              cash_folder='C:/Users/Julia/Documents/NSS_lab/fastnet/examples/info_log/mammonth_fix',
                                  graph_file='final_graph.pkl')

final_individ

euclid_dists = euclidean_distances(final_individ.source_data[final_individ.basis], final_individ.source_data[final_individ.basis])
matrix_connect = euclid_dists / np.max(euclid_dists)

plt.imshow(matrix_connect)
plt.colorbar()
plt.show()
plt.imshow(final_individ.matrix_connect)
plt.colorbar()
plt.show()
plt.imshow(matrix_connect - final_individ.matrix_connect)
plt.colorbar()
plt.show()



