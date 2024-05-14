import numpy as np
import topo as tp
from matplotlib import pyplot as plt

nodes_data = np.random.random((10, 10))


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)
