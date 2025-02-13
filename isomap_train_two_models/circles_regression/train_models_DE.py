import os,sys
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from torch import float32, nn, optim, float64
from torch.utils.data import TensorDataset, DataLoader

from scipy.optimize import differential_evolution

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps

device = 'cuda'

def plot_train_projection(train_points, reproj_points, predicted_classes, loss_value, filename):
    reproj_points = reproj_points.cpu().detach().numpy()
    try:
        train_points = train_points.cpu().detach().numpy()
        predicted_classes = predicted_classes.cpu().detach().numpy()
    except Exception as e:
        pass
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(train_points[:, 1], train_points[:, 0], c=predicted_classes)
    axs[0].set_title('Euclidean train classes')
    axs[1].scatter(reproj_points[:, 1], reproj_points[:, 0], c=predicted_classes)
    axs[1].set_title('Reprojected train classes')

    fig.suptitle(f'NN transformed: Train BCE={loss_value}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    #plt.show()


def to_polar(X):
    r = X[:, 0] ** 2 + X[:, 1] ** 2
    phi = torch.arctan(X[:, 1] / X[:, 0])
    polar = torch.stack((r, phi), axis=1)
    return polar

def generate_dataset():
    n_samples = 1000
    xs = np.random.uniform(low=-1, high=1, size=n_samples)
    ys = np.random.uniform(low=-1, high=1, size=n_samples)
    points = np.vstack((xs, ys)).T

    colors = np.array([(abs(point[0])+abs(point[1]))/2 for point in points])

    '''plt.scatter(points[:, 1], points[:, 0], c=colors)
    plt.colorbar()
    plt.show()'''

    X=points
    y=colors

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


train_features, test_features, validation_features, train_target, test_target, validation_target = generate_dataset()
train_features = torch.tensor(train_features, dtype=float32)
train_target = torch.tensor(train_target, dtype=float32)
test_target = torch.tensor(test_target, dtype=float32)

model_seq = [nn.Linear(train_features.size(1), 512, dtype=float32),
                                 nn.Linear(512, 256, dtype=float32),
                                 nn.Linear(256, 64, dtype=float32),
                                 nn.Linear(64, 1, dtype=float32)]

dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)
pts, reduced_dist = reduce_dist_fps(dist_train, 250)

reduced_train_target = train_target[pts]
reduced_train_features = train_features[pts]

test_dist = torch.tensor(pairwise_distances(test_features, reduced_train_features.cpu().detach().numpy()), dtype=float32).to(device)

dist_train_new = torch.zeros((train_features.shape[0], train_features[pts].shape[0]),device=device)
for i, x_train in enumerate(train_features):
    dist_train_new[i, :] = torch.linalg.norm(train_features[pts] - x_train, axis=1)
isomap_model = IsomapNN(reduced_dist)
isomap_model.to(device)

isomap_criterion = nn.L1Loss()
lr = 0.01
isomap_optimizer = optim.AdamW(isomap_model.parameters(), lr=lr)

working_folder = datetime.now().strftime('isomap_train_%Y%m%d_%H.%M')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)
working_folder = f'{os.getcwd()}/{working_folder}'

epochs = 1500
task_epochs = 5000
save_each = 200
losses = []

best_lost = np.inf
best_isomap_model = None

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def fitness(vect, train_dist=None):

    vect = torch.tensor(vect,dtype=torch.float32)

    tril_indices = torch.tril_indices(row=250, col=250, offset=0)

    reduced_dist = torch.zeros((250,250),dtype=torch.float32)

    reduced_dist[tril_indices[0], tril_indices[1]] = vect

    isomap_model = IsomapNN(reduced_dist)
    isomap_model.to(device)
    target = train_target.to(device)
    reduced_target = reduced_train_target.to(device)
    reproj_features = isomap_model().to(float32)
    with torch.no_grad():
        features = isomap_model.transform(dist_train_new)
        # ИНИЦИАЛИЗАЦИЯ ВТОРОЙ МОДЕЛИ
    task_model = nn.Sequential(*model_seq).to(device)
    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
    task_criterion = nn.L1Loss()
    early_stopper = EarlyStopper(patience=10, min_delta=1e-5)
    for ep in range(task_epochs):
        task_optim.zero_grad()
        out = task_model(features)
        task_loss = task_criterion(out, target.reshape_as(out))
        #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
        if early_stopper.early_stop(task_loss):             
            break
        task_loss.backward()
        task_optim.step()

    output = task_model(reproj_features)
    loss = isomap_criterion(output.to(torch.float32), reduced_target.reshape_as(output).to(torch.float32)).item()
    return loss

tril_indices = torch.tril_indices(row=250, col=250, offset=0)

vect = reduced_dist[tril_indices[0], tril_indices[1]].float()

#print(fitness(vect, train_dist = dist_train_new))

import cma

sigma0 = 1

xopt, es = cma.fmin2(fitness, vect, sigma0,{'bounds': [int(vect.shape[0])*[0], None]})

# for epoch in range(epochs):
#     epoch_losses = []
#     target = train_target.to(device)
#     reduced_target = reduced_train_target.to(device)

#     reproj_features = isomap_model().to(float32)
#     with torch.no_grad():
#         features = isomap_model.transform(dist_train_new)

#     # ИНИЦИАЛИЗАЦИЯ ВТОРОЙ МОДЕЛИ
#     task_model = nn.Sequential(*model_seq).to(device)
#     task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
#     task_criterion = nn.L1Loss()
#     for ep in range(task_epochs):
#         task_optim.zero_grad()
#         out = task_model(features)
#         task_loss = task_criterion(out, target.reshape_as(out))
#         #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
#         task_loss.backward()
#         task_optim.step()

#     output = task_model(reproj_features)
#     loss = isomap_criterion(output.to(torch.float32), reduced_target.reshape_as(output).to(torch.float32))
#     epoch_losses.append(loss.item())


#     if loss > 0.12:
#         for g in isomap_optimizer.param_groups:
#             g['lr'] = 0.01
#             lr = 0.01
#     if 0.12 >= loss >= 0.01:
#         for g in isomap_optimizer.param_groups:
#             g['lr'] = 0.0001
#             lr = 0.0001
#     if loss <= 0.01:
#         for g in isomap_optimizer.param_groups:
#             g['lr'] = 0.00001
#             lr = 0.00001

#     losses.append(np.mean(epoch_losses))
#     if losses[-1] < best_lost:
#         best_isomap_model = isomap_model
#         best_lost = losses[-1]
#         plot_train_projection(reduced_train_features, reproj_features, output, losses[-1], f'{working_folder}/{epoch}.png')
#         #torch.save(task_model.state_dict(), f'{working_folder}/best_task_model.pt')
#         best_reproj_points = reproj_features

#         reproj_features2 = best_isomap_model.transform(test_dist)
#         plot_train_projection(test_features, reproj_features2, test_target, losses[-1], f'{working_folder}/{epoch}_test.png')

#     loss.backward()
#     isomap_optimizer.step()

#     print(f'epoch {epoch}/{epochs}, lr={lr},  loss={losses[-1]}')


#     if epoch % save_each == 0:
#         plt.plot(np.arange(len(losses)), losses, label='Train')
#         plt.title('Convergence plot')
#         plt.ylabel('Loss')
#         plt.xlabel('Epochs')
#         plt.axhline(best_lost, c='r', linestyle='dashed')
#         plt.annotate(str(round(best_lost, 6)), (0, best_lost), c='r')
#         plt.legend()
#         plt.tight_layout()
#         #plt.yscale('log')
#         plt.savefig(f'{working_folder}/isomap_model_convergence.png')
#         plt.show()


# torch.save(best_isomap_model.state_dict(), f'{working_folder}/isomap_model.pt')

# plt.plot(np.arange(len(losses)), losses, label='Train')
# plt.title('Convergence plot')
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.axhline(best_lost, c='r', linestyle='dashed')
# plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
# plt.legend()
# plt.tight_layout()
# #plt.yscale('log')
# plt.savefig(f'{working_folder}/isomap_model_convergence.png')
# plt.show()


# test_proj_points = best_isomap_model.transform(test_dist)

# train_reproj_points = best_isomap_model.transform(dist_train_new)

# # инициализация модели для теста
# task_model = nn.Sequential(*model_seq).to(device)
# task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
# task_criterion = nn.BCELoss()
# for ep in range(task_epochs):
#     task_optim.zero_grad()
#     out = task_model(features)
#     task_loss = task_criterion(out, target.reshape_as(out))
#     #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
#     task_loss.backward()
#     task_optim.step()


# train_output = task_model(train_reproj_points.to(float32))
# output = task_model(test_proj_points.to(float32))

# test_proj_points = test_proj_points.cpu().detach().numpy()
# train_proj_points = train_reproj_points.cpu().detach().numpy()

# score = nn.L1Loss()
# train_acc = score(train_output, train_target.reshape_as(train_output).to(device)).item()
# test_acc = score(output, test_target.reshape_as(output).to(device)).item()

# fig, axs = plt.subplots(3, 2, figsize=(10, 10))
# axs[0, 0].scatter(test_proj_points[:, 1], test_proj_points[:, 0], c=test_target)
# axs[0, 0].set_title('Test: Reprojected target classes')
# axs[0, 1].scatter(test_proj_points[:, 1], test_proj_points[:, 0], c=output.cpu().detach().numpy())
# axs[0, 1].set_title('Test: Reprojected predicted classes')
# axs[1, 0].scatter(test_features[:, 1], test_features[:, 0], c=test_target)
# axs[1, 0].set_title('Test: Euclidean target classes')
# axs[1, 1].scatter(test_features[:, 1], test_features[:, 0], c=output.cpu().detach().numpy())
# axs[1, 1].set_title('Test: Euclidean predicted classes')

# axs[2, 0].scatter(train_features[:, 1], train_features[:, 0], c=train_output.cpu().detach().numpy())
# axs[2, 0].set_title('Train: Euclidean classes')
# axs[2, 1].scatter(train_proj_points[:, 1], train_proj_points[:, 0], c=train_output.cpu().detach().numpy())
# axs[2, 1].set_title('Train: Reprojected classes')

# fig.suptitle(f'NN transformed: Train L1={train_acc}, Test L1={test_acc}')
# plt.tight_layout()
# plt.savefig(f'{working_folder}/best_graph_prediction.png')
# plt.show()

# def to_polar(X):
#   r=X[:,0]**2+X[:,1]**2
#   phi=torch.arctan(X[:,1]/X[:,0])
#   polar=torch.stack((r,phi),axis=1)
#   return polar


# x = np.linspace(-1, 1, 10)
# y = np.linspace(-1, 1, 10)
# X, Y = np.meshgrid(x, y)
# grid = np.vstack([Y.ravel(), X.ravel()]).T

# plt.scatter(grid[:, 1], grid[:, 0])
# plt.title('Grid in euclidean coordinates')
# plt.show()

# polar_grid_features = to_polar(torch.tensor(grid))
# plt.scatter(polar_grid_features[:, 1], polar_grid_features[:, 0])
# plt.title('Grid in polar coordinates')
# plt.show()

# grid_dist = pairwise_distances(grid, train_features[pts])

# proj_grid_features = best_isomap_model.transform(torch.tensor(grid_dist, dtype=float32).to(device)).cpu().detach().numpy()

# plt.scatter(proj_grid_features[:, 1], proj_grid_features[:, 0])
# plt.title('Grid in transformed coordinates')
# plt.savefig(f'{working_folder}/grid_with_isomap.png')
# plt.show()

