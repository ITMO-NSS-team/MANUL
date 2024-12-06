import matplotlib.pyplot as plt
import pandas as pd

from evolution.IndividStructures import DataStructureGraph


def load_data():
    df = pd.read_csv('data/kitti.csv', delimiter=';')
    data = df.to_numpy()
    return data

data = load_data()
base_individ = DataStructureGraph(data=data,
                                      cache_folder=None,
                                      n_neighbors=2,
                                      epsilon_neighborhood=1)
base_individ.show_3d()
base_individ.show_2d()