from datetime import datetime
import numpy as np
from Eva.Evolution import Evolution


device = 'cuda'

def circles_2d(n_samples=1000):
    xs = np.random.uniform(low=-1, high=1, size=n_samples)
    ys = np.random.uniform(low=-1, high=1, size=n_samples)
    points = np.vstack((xs, ys)).T
    colors = np.array([(abs(point[0]) + abs(point[1])) / 2 for point in points])
    return points, colors

exp_folder = f'circles_{datetime.now().strftime("%d%m%Y-%H.%M")}'
data, labels = circles_2d(100)
latent_len = 2
train_features = data
train_targets = labels

evolution = Evolution(train_features=train_features,
                      train_targets=train_targets,
                      latent_len=latent_len,
                      population_size=20,
                      iterations=1000,
                      logs_folder=exp_folder,
                      edges_weight_mutation=True,
                      edges_mutation=False)

evolution.run()
evolution.plot_evolution_fitnesses()
evolution.plot_evolution_fitnesses(reverse=True, save_path=f'{exp_folder}/conv.png')
