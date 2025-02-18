import os,sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchvision import datasets

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))


from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps

from sklearn.decomposition import PCA

device = 'cuda'

def plot_train_projection(train_points, reproj_points, predicted_classes, real_classes, loss_value, acc_value, filename):
    train_points = train_points.cpu().detach().numpy()
    reproj_points = reproj_points.cpu().detach().numpy()

    train_points_2d = PCA(n_components=2).fit_transform(train_points)
    reproj_points_2d = PCA(n_components=2).fit_transform(reproj_points)

    predicted_classes = torch.argmax(predicted_classes, dim=1).cpu().detach().numpy()
    real_classes = torch.argmax(real_classes, dim=1).cpu().detach().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    axs[0].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=real_classes)
    axs[0].set_title('Euclidean (target)')
    axs[1].scatter(reproj_points_2d[:, 1], reproj_points_2d[:, 0], c=predicted_classes)
    axs[1].set_title('Reprojected (predicted)')
    axs[2].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=predicted_classes)
    axs[2].set_title('Euclidean (predicted)')

    fig.suptitle(f'MNIST\n CrossEntropyLoss={loss_value}, '
                 f'accuracy={acc_value}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def accuracy(predicted, target):
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc


def init_data(train_size=30000, validation_size=10000, test_size=10000):
    dataset = datasets.MNIST('../mnist_data', train=True, download=False)
    data = dataset.train_data.numpy() / 255
    labels = dataset.train_labels.numpy()
    data = np.expand_dims(data, axis=1)
    # CROP TRAIN SET
    train_labels = labels[:train_size]
    train_data = data[:train_size, :]
    # CROP VAL SET
    val_labels = labels[train_size:train_size+validation_size]
    val_data = data[train_size:train_size+validation_size, :]
    # CROP TEST SET
    test_labels = labels[:test_size]
    test_data = data[:test_size, :]
    # TRAIN LABELS TO PROBS
    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1
    # VAL LABELS TO PROBS
    val_labels_log = np.zeros((val_labels.shape[0], 10))
    for i in range(val_labels_log.shape[0]):
        val_labels_log[i][val_labels[i]] = 1
    # TEST LABELS TO PROBS
    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1
    return train_data, train_labels_log, val_data, val_labels_log, test_data, test_labels_log



train_features, train_target, val_features, val_target, test_features, test_target = init_data()
# RAVEL DATA
train_features = train_features.reshape(train_features.shape[0],
                                            train_features.shape[1] *
                                            train_features.shape[2] *
                                            train_features.shape[3])
test_features = test_features.reshape(test_features.shape[0],
                                            test_features.shape[1] *
                                            test_features.shape[2] *
                                            test_features.shape[3])
val_features = val_features.reshape(val_features.shape[0],
                                            val_features.shape[1] *
                                            val_features.shape[2] *
                                            val_features.shape[3])

retain_points = 1000

dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)


# SELECT SPARSE POINTS
pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)
reduced_train_target = torch.tensor(train_target[pts], dtype=float32)
reduced_train_features = torch.tensor(train_features[pts], dtype=float32)



test_dist = torch.tensor(pairwise_distances(test_features, reduced_train_features), dtype=float32).to(device)
val_dist = torch.tensor(pairwise_distances(val_features, reduced_train_features), dtype=float32).to(device)

dist_train_new = torch.tensor(pairwise_distances(train_features, train_features[pts]), dtype=float32).to(device)

latent_len = 8
isomap_model = IsomapNN(reduced_dist, n_components=latent_len)

isomap_model.to(device)

model_seq = [nn.Linear(latent_len, 512, dtype=float32),
                                 nn.Linear(512, 256, dtype=float32),
                                 nn.Linear(256, 64, dtype=float32),
                                 nn.Linear(64, 10, dtype=float32),  # 10 classes
                                 ]
print(model_seq)

train_features = torch.tensor(train_features, dtype=float32).to(device)
train_target = torch.tensor(train_target, dtype=float32).to(device)
val_features = torch.tensor(val_features, dtype=float32).to(device)
val_target = torch.tensor(val_target, dtype=float32).to(device)
test_features = torch.tensor(test_features, dtype=float32).to(device)
test_target = torch.tensor(test_target, dtype=float32).to(device)

isomap_epochs = 2000
task_epochs = 300
save_each = 100

working_folder = datetime.now().strftime(f'ICML_RESULTS/mnist_fnn_({latent_len})%Y%m%d_%H.%M')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)
    #os.makedirs(f'{working_folder}/optimization')

working_folder = f'{os.getcwd()}/{working_folder}'

lr = 1e-2
isomap_optim = torch.optim.Adam(params=isomap_model.parameters(), lr=lr)
isomap_criterion = nn.CrossEntropyLoss()
losses = []
val_losses = []
best_loss = np.inf
best_val_loss = np.inf
best_isomap_model = None
patience = 0

for epoch in range(isomap_epochs):
    isomap_optim.zero_grad()
    target = train_target.to(device)
    reduced_target = reduced_train_target.to(device)
    reproj_features = isomap_model().to(float32)

    with torch.no_grad():
        features = isomap_model.transform(dist_train_new)
        proj_val_features = isomap_model.transform(val_dist)

    task_model = nn.Sequential(*model_seq).to(device)
    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
    task_criterion = nn.CrossEntropyLoss()

    task_losses = []
    for ep in range(task_epochs):
        task_optim.zero_grad()
        out = task_model(features)
        task_loss = task_criterion(out.reshape_as(train_target), train_target)
        #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
        task_loss.backward()
        task_optim.step()
        task_losses.append(task_loss.item())


    output = task_model(reproj_features)
    isomap_loss = isomap_criterion(output.to(torch.float32), reduced_target.reshape_as(output).to(torch.float32))
    losses.append(isomap_loss.item())

    validation_output = task_model(proj_val_features)
    val_loss = isomap_criterion(validation_output.to(torch.float32), val_target.reshape_as(validation_output).to(torch.float32))
    val_losses.append(val_loss.item())

    isomap_loss.backward()
    isomap_optim.step()

    print(f'epoch {epoch}/{isomap_epochs}, lr={lr},  loss={losses[-1]}, val_loss={val_losses[-1]}')
    #torch.save(isomap_model.distances_matrix, f'{working_folder}/optimization/distance_matrix_{epoch}.pt')
    

    #if isomap_loss.item() > 0.15:
    #    for g in isomap_optim.param_groups:
    #        g['lr'] = 0.01
    #        lr = 0.01
    #if 0.15 >= isomap_loss.item() >= 0.01:
    #    for g in isomap_optim.param_groups:
    #        g['lr'] = 0.001
    #        lr = 0.001
    #if isomap_loss.item() <= 0.01:
    #    for g in isomap_optim.param_groups:
    #        g['lr'] = 0.0001
    #        lr = 0.0001

    if losses[-1] < best_loss:
        #best_isomap_model = isomap_model
        best_loss = losses[-1]

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        best_isomap_model = isomap_model
        if epoch % 10 == 0 or epoch == isomap_epochs-1:
            train_acc = accuracy(torch.argmax(output, dim=1), torch.argmax(reduced_target, dim=1))
            val_accuracy = accuracy(torch.argmax(validation_output, dim=1), torch.argmax(val_target, dim=1))

            plot_train_projection(val_features, proj_val_features, validation_output, val_target, val_losses[-1], val_accuracy,
                                  f'{working_folder}/{epoch}_validation.png')
            plot_train_projection(reduced_train_features, reproj_features, output, reduced_target, losses[-1], train_acc, f'{working_folder}/{epoch}_train.png')
            #torch.save(best_isomap_model.state_dict(), f'{working_folder}/isomap_model.pt')

        if epoch == 0:
            proj_test = best_isomap_model.transform(test_dist)
            test_out = task_model(proj_test)
            test_out = test_out.reshape_as(test_target)
            test_CEL = task_criterion(test_out, test_target)
            test_acc = accuracy(torch.argmax(test_out, dim=1), torch.argmax(test_target, dim=1))
            df = pd.DataFrame()
            df['CEL_train'] = [losses[-1]]
            df['CEL_validation'] = [val_losses[-1]]
            df['CEL_test'] = [test_CEL.cpu().detach().numpy()]
            df['acc_train'] = [train_acc.cpu().detach().numpy()]
            df['acc_validation'] = [val_accuracy.cpu().detach().numpy()]
            df['acc_test'] = [test_acc.cpu().detach().numpy()]
            df.to_csv(f'{working_folder}/isomap_raw_metrics.csv', index=False)


    if epoch % save_each == 0:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(np.arange(len(losses)), losses)
        axs[0].set_title('Train')
        axs[0].axhline(best_loss, c='r', linestyle='dashed')
        axs[0].annotate(str(round(best_loss, 4)), (0, best_loss), c='r')
        axs[0].set_ylabel('Loss')
        axs[0].set_xlabel('Epochs')
        axs[1].plot(np.arange(len(val_losses)), val_losses, c='orange')
        axs[1].set_title('Validation')
        axs[1].axhline(best_val_loss, c='green', linestyle='dashed')
        axs[1].annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epochs')
        plt.suptitle('Convergence plot')
        plt.tight_layout()
        plt.savefig(f'{working_folder}/convergence.png')
        #plt.show()


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(np.arange(len(losses)), losses)
axs[0].set_title('Train')
axs[0].axhline(best_loss, c='r', linestyle='dashed')
axs[0].annotate(str(round(best_loss, 4)), (0, best_loss), c='r')
axs[0].set_ylabel('Loss')
axs[0].set_xlabel('Epochs')
axs[1].plot(np.arange(len(val_losses)), val_losses, c='orange')
axs[1].set_title('Validation')
axs[1].axhline(best_val_loss, c='green', linestyle='dashed')
axs[1].annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')
axs[1].set_ylabel('Loss')
axs[1].set_xlabel('Epochs')
plt.suptitle('Convergence plot')
plt.tight_layout()
plt.savefig(f'{working_folder}/convergence.png')
plt.show()

task_model = nn.Sequential(*model_seq).to(device)
task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
task_criterion = nn.CrossEntropyLoss()

task_losses = []
for ep in range(task_epochs):
    task_optim.zero_grad()
    out = task_model(features)
    task_loss = task_criterion(out.reshape_as(train_target), train_target)
    #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
    task_loss.backward()
    task_optim.step()
    task_losses.append(task_loss.item())

train_acc = accuracy(torch.argmax(out, dim=1), torch.argmax(train_target, dim=1))

proj_val = best_isomap_model.transform(val_dist)
val_out = task_model(proj_val)
val_out = val_out.reshape_as(val_target)
val_CEL = task_criterion(val_out, val_target)
val_acc = accuracy(torch.argmax(val_out, dim=1), torch.argmax(val_target, dim=1))

proj_test = best_isomap_model.transform(test_dist)
test_out = task_model(proj_test)
test_out = test_out.reshape_as(test_target)
test_CEL = task_criterion(test_out, test_target)
print(f'Test cross entropy {test_CEL}')
test_acc = accuracy(torch.argmax(test_out, dim=1), torch.argmax(test_target, dim=1)).cpu().detach().numpy()
print(f'Test accuracy {test_acc}')


plot_train_projection(test_features, proj_test, test_out, test_target, test_CEL, test_acc, f'{working_folder}/test.png')


df = pd.DataFrame()
df['CEL_train'] = [task_loss.item()]
df['CEL_validation'] = [val_CEL.cpu().detach().numpy()]
df['CEL_test'] = [test_CEL.cpu().detach().numpy()]
df['acc_train'] = [train_acc.cpu().detach().numpy()]
df['acc_validation'] = [val_acc.cpu().detach().numpy()]
df['acc_test'] = [test_acc]
df.to_csv(f'{working_folder}/isomap_optimized_metrics.csv', index=False)
