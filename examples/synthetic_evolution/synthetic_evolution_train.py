from datetime import datetime


from evolution.Evolution_new import Evolution
from examples.data.synthetic_data_generation import geometries

device = 'cuda'
for geometry in ['swiss_roll']:
    data, labels = geometries[geometry][0]()
    latent_len = geometries[geometry][1]
    train_features = data
    train_targets = labels

    evolution = Evolution(train_features=train_features,
                          train_targets=train_targets,
                          latent_len=latent_len,
                          population_size=10,
                          iterations=200,
                          logs_folder=f'{geometry}_{datetime.now().strftime("%d%m%Y-%H.%M")}')

    evolution.run()
    evolution.plot_evolution_fitnesses()
    evolution.plot_evolution_fitnesses(reverse=True)
    for individ in evolution.population.individs_pool:
        individ.visualize()
