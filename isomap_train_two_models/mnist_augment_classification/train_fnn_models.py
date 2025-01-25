import os
from datetime import datetime
from scipy import ndimage
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from torch import float32, nn
from torchvision import datasets

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps, plot_points_with_PCA

from sklearn.decomposition import PCA

device = 'cuda'

def plot_train_projection(train_points, reproj_points, predicted_classes, labels, loss_value, filename):
    labels_colors = plt.cm.Reds(labels/90)
    train_points = train_points.cpu().detach().numpy()
    reproj_points = reproj_points.cpu().detach().numpy()

    train_points_2d = PCA(n_components=2).fit_transform(train_points)
    reproj_points_2d = PCA(n_components=2).fit_transform(reproj_points)

    predicted_classes = torch.argmax(predicted_classes, dim=1).cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=predicted_classes, edgecolors=labels_colors)
    axs[0].set_title('Euclidean train classes')
    axs[1].scatter(reproj_points_2d[:, 1], reproj_points_2d[:, 0], c=predicted_classes, edgecolors=labels_colors)
    axs[1].set_title('Reprojected train classes')

    fig.suptitle(f'PCA MNIST Transform\nTrain CrossEntropyLoss={loss_value}')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def accuracy(predicted, target):
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc


def init_data(train_size=20000, test_size=10000):
    dataset = datasets.MNIST('../mnist_data', train=True, download=False)
    #dataset = datasets.FashionMNIST('data_fashion', train=True, download=True)
    train_data = dataset.train_data.numpy() / 255
    train_labels = dataset.train_labels.numpy()
    # CROP TRAIN SET
    train_labels = train_labels[:train_size]
    train_data = train_data[:train_size, :]
    # INIT TEST
    test_data = dataset.test_data.numpy() / 255
    test_labels = dataset.test_labels.numpy()
    # TRAIN LABELS TO PROBS
    train_labels_log = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels_log.shape[0]):
        train_labels_log[i][train_labels[i]] = 1
    # CROP TEST SET
    test_labels = test_labels[:test_size]
    test_data = test_data[:test_size, :]
    # TEST LABELS TO PROBS
    test_labels_log = np.zeros((test_labels.shape[0], 10))
    for i in range(test_labels_log.shape[0]):
        test_labels_log[i][test_labels[i]] = 1
    return train_data, train_labels_log, test_data, test_labels_log

def create_dataset(x_data, y_data):
    x_result = []
    y_result = []
    ang_result = []

    def append_data(newx, newy, ang):
        x_result.append(newx)
        y_result.append(newy)
        ang_result.append(ang)

    for i in range(x_data.shape[0]):
        image = x_data[i]
        label = y_data[i]
        append_data(image, label, 0)

        height, width = image.shape[:2]

        for ange in np.arange(30, 95, 30):
            rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), ange, 1)
            rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
            append_data(rotated_image, label, ange)

    x_result = np.array(x_result)
    x_result = np.expand_dims(x_result, axis=1)
    y_result = np.array(y_result)
    ang_result = np.array(ang_result)

    return x_result, y_result, ang_result


N = 5000
train_features, train_target, _, _ = init_data(train_size=N,)
train_features, train_target, train_ang = create_dataset(train_features, train_target)
#test_features, test_target, test_ang = create_dataset(test_features, test_target)

# RAVEL DATA
train_features = train_features.reshape(train_features.shape[0],
                                            train_features.shape[1] *
                                            train_features.shape[2] *
                                            train_features.shape[3])
'''test_features = test_features.reshape(test_features.shape[0],
                                            test_features.shape[1] *
                                            test_features.shape[2] *
                                            test_features.shape[3])'''


retain_points = 1000

dist_train = torch.tensor(pairwise_distances(train_features, train_features), dtype=float32)
# SELECT SPARSE POINTS
pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)
reduced_train_target = torch.tensor(train_target[pts], dtype=float32)
reduced_train_features = torch.tensor(train_features[pts], dtype=float32)
reduced_train_ang = train_ang[pts]

plot_points_with_PCA(train_features, train_target)
plot_points_with_PCA(reduced_train_features, reduced_train_target)

'''plt.scatter(train_features[:, 1], train_features[:, 0], c=train_target)
plt.title('Full dataset')
plt.show()
plt.scatter(reduced_train_features[:, 1], reduced_train_features[:, 0], c=reduced_train_target)
plt.title('Reduced dataset')
plt.show()'''

#test_dist = torch.tensor(pairwise_distances(test_features, reduced_train_features), dtype=float32).to(device)

dist_train_new = torch.tensor(pairwise_distances(train_features, train_features[pts]), dtype=float32).to(device)

latent_len = 15
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
#test_features = torch.tensor(test_features, dtype=float32).to(device)
#test_target = torch.tensor(test_target, dtype=float32).to(device)

isomap_epochs = 5000
task_epochs = 300
save_each = 100

working_folder = datetime.now().strftime(f'mnist_aug_{N}_%Y%m%d_%H.%M')
if not os.path.exists(working_folder):
    os.makedirs(working_folder)
working_folder = f'{os.getcwd()}/{working_folder}'

lr = 0.001
isomap_optim = torch.optim.Adam(params=isomap_model.parameters(), lr=lr)
isomap_criterion = nn.CrossEntropyLoss()
losses = []
best_lost = np.inf
best_isomap_model = None

for epoch in range(isomap_epochs):
    isomap_optim.zero_grad()
    reduced_target = reduced_train_target.to(device)
    reproj_features = isomap_model().to(float32)

    with torch.no_grad():
        features = isomap_model.transform(dist_train_new)

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

    #plt.plot(np.arange(len(task_losses)), task_losses)
    #plt.show()

    output = task_model(reproj_features)
    isomap_loss = isomap_criterion(output.to(torch.float32), reduced_target.reshape_as(output).to(torch.float32))
    losses.append(isomap_loss.item())

    isomap_loss.backward()
    isomap_optim.step()

    print(f'epoch {epoch}/{isomap_epochs}, lr={lr},  loss={losses[-1]}')

    if losses[-1] < best_lost:
        best_isomap_model = isomap_model
        best_lost = losses[-1]
        plot_train_projection(reduced_train_features,
                              reproj_features,
                              output,
                              reduced_train_ang,
                              losses[-1],
                              f'{working_folder}/{epoch}.png')
        torch.save(best_isomap_model.state_dict(), f'{working_folder}/isomap_model.pt')


    if epoch % save_each == 0:
        plt.plot(np.arange(len(losses)), losses)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.title('Convergence plot')
        plt.axhline(best_lost, c='r', linestyle='dashed')
        plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
        plt.tight_layout()
        plt.savefig(f'{working_folder}/convergence.png')
        plt.show()

plt.plot(np.arange(len(losses)), losses)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Convergence plot')
plt.axhline(best_lost, c='r', linestyle='dashed')
plt.annotate(str(round(best_lost, 4)), (0, best_lost), c='r')
plt.tight_layout()
#plt.yscale('log')
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


#proj_test = best_isomap_model.transform(test_dist)
#test_out = task_model(proj_test)
#test_out = test_out.reshape_as(test_target)
#test_CEL = task_criterion(test_out, test_target)
#print(f'Test cross entropy {test_CEL}')

#test_out = torch.argmax(test_out, dim=1)
#test_target = torch.argmax(test_target, dim=1)

#test_acc = accuracy(test_out, test_target)
#print(f'Test accuracy {test_acc}')


