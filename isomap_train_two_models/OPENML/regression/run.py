import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import os
from datetime import datetime

import openml
from torch import float32, nn

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.OPENML.architectures import regres_fnn
from isomap_train_two_models.utils import reduce_dist_fps

device = 'cuda'

openml.config.server = "http://145.38.195.79/api/v1/xml"

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def mae(prediction, target):
    return torch.mean(abs(prediction - target))


def plot_points_with_PCA(points, labels, save_dir, name):
    points_2d = PCA(n_components=2).fit_transform(points.cpu().detach().numpy())
    sc = plt.scatter(points_2d[:, 1], points_2d[:, 0], c=labels.cpu().detach().numpy())
    plt.colorbar(sc)
    plt.savefig(f'{save_dir}/{name}.png')
    #plt.show()
    plt.close()


def split_train_test(dataset, target_name):
    y = dataset[target_name].to_numpy()
    X = dataset[dataset.columns.drop(target_name)].to_numpy()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.25)
    return X_train, X_test, X_validation, y_train, y_test, y_validation


def run_openml_regression(n_times):
    datasets_folder = '../datasets/regression'
    for file in os.listdir(datasets_folder):
        if file not in os.listdir('ICML_RESULTS'):
            dataset_df = pd.read_csv(f'{datasets_folder}/{file}')
            dataset_name = file
            target_name = dataset_df.columns[-1]
            try:
                # dataset_df = dataset_df.dropna()
                for column in dataset_df.columns:
                    if dataset_df[column].dtype.name in ['object', 'category', 'bool']:
                        try:
                            dataset_df[column] = dataset_df[column].astype(int)
                        except Exception as e:
                            try:
                                encoder = LabelEncoder()
                                encoder.fit_transform(dataset_df[column].to_frame())
                                dataset_df[column] = encoder.transform(dataset_df[column].to_frame())
                            except Exception as e:
                                pass
                dataset_df = dataset_df.apply(pd.to_numeric, errors='coerce')
                dataset_df = dataset_df.dropna()
                for col in dataset_df.columns:
                    try:
                        dataset_df[col] = dataset_df[col] / dataset_df[col].max()
                    except Exception as e:
                        print(e)
                        pass
                dataset_df = dataset_df.dropna()
                if len(dataset_df) < 100:
                    print(f'skip {dataset_name} - {len(dataset_df)}')
                    continue

                ds_folder = f'ICML_RESULTS/{dataset_name}'
                if not os.path.exists(ds_folder):
                    os.makedirs(ds_folder)

                dataset_df.to_csv(f'{ds_folder}/{dataset_name}', index=False)

                for n in range(n_times):
                    working_folder = datetime.now().strftime(f'{ds_folder}/%Y%m%d_%H.%M')
                    if not os.path.exists(working_folder):
                        os.makedirs(working_folder)
                        os.makedirs(f'{working_folder}/optimization')

                    X_train, X_test, X_validation, y_train, y_test, y_validation = split_train_test(dataset_df, target_name)
                    max_target = np.max(y_train)
                    y_train = y_train/max_target
                    y_validation = y_validation/max_target
                    y_test = y_test/max_target

                    dist_train = torch.tensor(pairwise_distances(X_train, X_train), dtype=float32)
                    max_val = torch.max(dist_train)
                    dist_train = dist_train / max_val
                    # SELECT SPARSE POINTS
                    if X_train.shape[0] > 1000:
                        retain_points = 200
                    else:
                        retain_points = X_train.shape[0]
                    pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)

                    dist_train_new = torch.tensor(pairwise_distances(X_train, X_train[pts]), dtype=float32).to(
                        device) / max_val
                    val_dist = torch.tensor(pairwise_distances(X_validation, X_train[pts]), dtype=float32).to(
                        device) / max_val

                    X_train = torch.tensor(X_train, dtype=float32).to(device)
                    y_train = torch.tensor(y_train, dtype=float32).to(device)
                    y_validation = torch.tensor(y_validation, dtype=float32).to(device)
                    y_test = torch.tensor(y_test, dtype=float32).to(device)

                    reduced_train_target = torch.tensor(y_train[pts], dtype=float32)
                    reduced_train_features = torch.tensor(X_train[pts], dtype=float32)

                    plot_points_with_PCA(X_train, y_train, working_folder, 'raw_data_PCA')
                    plot_points_with_PCA(reduced_train_features, reduced_train_target, working_folder, 'reduced_data_PCA')

                    # reduce features into isomap is its extensive
                    if X_train.shape[-1] > 15:
                        latent_len = 15
                    else:
                        latent_len = X_train.shape[-1]

                    isomap_model = IsomapNN(reduced_dist, n_components=latent_len)
                    isomap_model.to(device)

                    isomap_epochs = 100
                    task_epochs = 300
                    isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.01)
                    isomap_criterion = nn.L1Loss()
                    validation_criterion = nn.L1Loss()
                    losses = []
                    val_losses = []
                    best_loss = np.inf
                    best_val_loss = np.inf
                    best_isomap_model = None

                    # ISOMAP TRAIN LOOP
                    for epoch in range(isomap_epochs):
                        reproj_features = isomap_model().to(float32)

                        plt.scatter(reproj_features.cpu().detach().numpy()[:, 1],
                                    reproj_features.cpu().detach().numpy()[:, 0],
                                    c=reduced_train_target.cpu().detach().numpy())
                        plt.show()

                        with torch.no_grad():
                            features = isomap_model.transform(dist_train_new)

                        task_model = regres_fnn(latent_len).to(device)
                        task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                        task_criterion = nn.L1Loss()

                        task_losses = []
                        for ep in range(task_epochs):
                            task_optim.zero_grad()
                            out = task_model(features)
                            task_loss = task_criterion(out, y_train)
                            #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                            task_loss.backward()
                            task_optim.step()
                            task_losses.append(task_loss.item())

                        output = task_model(reproj_features)
                        isomap_loss = isomap_criterion(output.to(torch.float32),
                                                       reduced_train_target.reshape_as(reduced_train_target).to(torch.float32))
                        losses.append(isomap_loss.item())

                        # VALIDATION
                        reproj_val_features = isomap_model.transform(val_dist)
                        val_output = task_model(reproj_val_features)
                        val_loss = validation_criterion(val_output, y_validation)
                        val_losses.append(val_loss.item())

                        if losses[-1] < best_loss:
                            best_loss = losses[-1]

                        if val_losses[-1] < best_val_loss:
                            best_val_loss = val_losses[-1]
                            best_isomap_model = isomap_model
                            # SAVE OPTIMIZATION PROCESS
                            np.save(f'{working_folder}/optimization/distance_matrix_{epoch}.npy',
                                    isomap_model.distances_matrix.cpu().detach().numpy())

                        isomap_loss.backward()
                        isomap_optim.step()
                        print(f'epoch {epoch}/{isomap_epochs},  loss={losses[-1]}, val_loss={val_losses[-1]}')

                    plt.figure()
                    plt.plot(np.arange(len(losses)), losses, label='Train')
                    plt.plot(np.arange(len(val_losses)), val_losses, label='Validation')
                    plt.title('Convergence plot')
                    plt.ylabel('Loss')
                    plt.xlabel('Epochs')
                    plt.axhline(best_val_loss, c='green', linestyle='dashed')
                    plt.axhline(best_loss, c='r', linestyle='dashed')
                    plt.annotate(str(round(best_val_loss, 4)), (0, best_val_loss), c='green')
                    plt.annotate(str(round(best_loss, 4)), (0, best_loss), c='r')
                    plt.legend()
                    plt.tight_layout()
                    plt.yscale('log')
                    plt.savefig(f'{working_folder}/isomap_model_convergence.png')
                    #plt.show()
                    plt.close()

                    # ISOMAP POINTS PROJECTION
                    val_proj_points = best_isomap_model.transform(val_dist)
                    train_reproj_points = best_isomap_model.transform(dist_train_new)
                    test_dist = torch.tensor(
                        pairwise_distances(X_test, reduced_train_features.cpu().detach().numpy()),
                        dtype=float32).to(device) / max_val
                    test_proj_points = best_isomap_model.transform(test_dist)

                    task_model = regres_fnn(latent_len).to(device)
                    task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                    task_criterion = nn.L1Loss()
                    for ep in range(task_epochs):
                        task_optim.zero_grad()
                        out = task_model(features)
                        task_loss = task_criterion(out, y_train)
                        # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                        task_loss.backward()
                        task_optim.step()

                    train_output = task_model(train_reproj_points.to(float32))[:, 0]
                    val_output = task_model(val_proj_points.to(float32))[:, 0]
                    test_output = task_model(test_proj_points.to(float32))[:, 0]

                    # METRICS CALCULATION
                    score = nn.L1Loss()
                    train_score = score(train_output, y_train).item()
                    val_score = score(val_output, y_validation).item()
                    test_score = score(test_output, y_test).item()

                    train_mae = mae(train_output, y_train).cpu().detach().numpy()
                    val_mae = mae(val_output, y_validation).cpu().detach().numpy()
                    test_mae = mae(test_output, y_test).cpu().detach().numpy()

                    df = pd.DataFrame()
                    df['L1_train'] = [train_score]
                    df['L1_validation'] = [val_score]
                    df['L1_test'] = [test_score]
                    df['MAE_train'] = [train_mae]
                    df['MAE_validation'] = [val_mae]
                    df['MAE_test'] = [test_mae]
                    df.to_csv(f'{working_folder}/metrics.csv', index=False)

                    train_proj_points = train_reproj_points.cpu().detach().numpy()
                    val_proj_points = val_proj_points.cpu().detach().numpy()
                    test_proj_points = test_proj_points.cpu().detach().numpy()

                    train_proj_points_2d = PCA(n_components=2).fit_transform(train_proj_points)
                    train_points_2d = PCA(n_components=2).fit_transform(X_train.cpu().detach().numpy())
                    val_proj_points_2d = PCA(n_components=2).fit_transform(val_proj_points)
                    val_points_2d = PCA(n_components=2).fit_transform(X_validation)
                    test_proj_points_2d = PCA(n_components=2).fit_transform(test_proj_points)
                    test_points_2d = PCA(n_components=2).fit_transform(X_test)

                    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
                    cs0 = axs[0].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=y_train.cpu().detach().numpy())
                    fig.colorbar(cs0, ax=axs[0])
                    axs[0].set_title('Euclidean train (target)')
                    cs1 = axs[1].scatter(train_proj_points_2d[:, 1], train_proj_points_2d[:, 0],
                                   c=train_output.cpu().detach().numpy())
                    fig.colorbar(cs1, ax=axs[1])
                    axs[1].set_title('Reprojected train (predicted)')
                    cs2 = axs[2].scatter(train_points_2d[:, 1], train_points_2d[:, 0], c=train_output.cpu().detach().numpy())
                    fig.colorbar(cs2, ax=axs[2])
                    axs[2].set_title('Euclidean train (predicted)')
                    fig.suptitle(f'Train set L1={train_score}, MAE={train_mae}')
                    plt.tight_layout()
                    plt.savefig(f'{working_folder}/train_inference.png')
                    plt.close()

                    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
                    cs0 = axs[0].scatter(val_points_2d[:, 1], val_points_2d[:, 0], c=y_validation.cpu().detach().numpy())
                    fig.colorbar(cs0, ax=axs[0])
                    axs[0].set_title('Euclidean validation (target)')
                    cs1 = axs[1].scatter(val_proj_points_2d[:, 1], val_proj_points_2d[:, 0],
                                   c=val_output.cpu().detach().numpy())
                    fig.colorbar(cs1, ax=axs[1])
                    axs[1].set_title('Reprojected validation (predicted)')
                    cs2 = axs[2].scatter(val_points_2d[:, 1], val_points_2d[:, 0], c=val_output.cpu().detach().numpy())
                    fig.colorbar(cs2, ax=axs[2])
                    axs[2].set_title('Euclidean validation (predicted)')
                    fig.suptitle(f'Validation set L1={train_score}, MAE={train_mae}')
                    plt.tight_layout()
                    plt.savefig(f'{working_folder}/validation_inference.png')
                    plt.close()

                    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
                    cs0 = axs[0].scatter(test_points_2d[:, 1], test_points_2d[:, 0], c=y_test.cpu().detach().numpy())
                    fig.colorbar(cs0, ax=axs[0])
                    axs[0].set_title('Euclidean test (target)')
                    cs1 = axs[1].scatter(test_proj_points_2d[:, 1], test_proj_points_2d[:, 0],
                                         c=test_output.cpu().detach().numpy())
                    fig.colorbar(cs1, ax=axs[1])
                    axs[1].set_title('Reprojected test (predicted)')
                    cs2 = axs[2].scatter(test_points_2d[:, 1], test_points_2d[:, 0], c=test_output.cpu().detach().numpy())
                    fig.colorbar(cs2, ax=axs[2])
                    axs[2].set_title('Euclidean test (predicted)')
                    fig.suptitle(f'Test set L1={train_score}, MAE={train_mae}')
                    plt.tight_layout()
                    plt.savefig(f'{working_folder}/test_inference.png')
                    plt.close()

                    torch.cuda.empty_cache()

            except KeyError as e:
                print(f'Skip {dataset_name}\n{e}')


run_openml_regression(1)
