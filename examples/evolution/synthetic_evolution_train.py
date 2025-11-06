from datetime import datetime

from Eva.Evolution import Evolution
from examples.data.synthetic_data_generation import geometries

device = 'cuda'
for geometry in geometries:
    exp_folder = f'{geometry}_{datetime.now().strftime("%d%m%Y-%H.%M")}'
    data, labels = geometries[geometry][0](100)
    latent_len = geometries[geometry][1]
    train_features = data
    train_targets = labels

    evolution = Evolution(train_features=train_features,
                          train_targets=train_targets,
                          latent_len=latent_len,
                          population_size=20,
                          iterations=100,
                          logs_folder=exp_folder,
                          edges_weight_mutation=True,
                          edges_mutation=False)

    evolution.run()
    evolution.plot_evolution_fitnesses()
    evolution.plot_evolution_fitnesses(reverse=True, save_path=f'{exp_folder}/conv.png')
