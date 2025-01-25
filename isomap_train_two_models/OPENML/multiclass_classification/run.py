import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import os
from datetime import datetime

from torch import float32, nn

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.OPENML.architectures import binary_fnn
from isomap_train_two_models.utils import reduce_dist_fps
import warnings
warnings.filterwarnings("ignore")
device = 'cuda'


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



def accuracy(predicted, target):
    predicted = torch.argmax(predicted, dim=1)
    acc = torch.sum(predicted == target)
    acc = acc / predicted.size(0)
    return acc

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


def run_openml_binary(n_times):
    datasets_folder = '../datasets/multiclass_classification'
    for file in os.listdir(datasets_folder):
        if file not in os.listdir('ICML_RESULTS'):
            dataset_df = pd.read_csv(f'{datasets_folder}/{file}')
            #dataset_name = ''.join(file.split('_')[1:])
            dataset_name = file
            target_name = dataset_df.columns[-1]
            try:
                #dataset_df = dataset_df.dropna()
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
                        dataset_df[col] = dataset_df[col]/dataset_df[col].max()
                    except Exception as e:
                        print(e)
                        pass
                dataset_df = dataset_df.dropna()
                if len(dataset_df) < 100:
                    print(f'skip {dataset_name} - {len(dataset_df)}')
                    continue

                try:
                    ds_folder = f'ICML_RESULTS/{dataset_name}'
                    if not os.path.exists(ds_folder):
                        os.makedirs(ds_folder)

                    dataset_df.to_csv(f'{ds_folder}/{dataset_name}', index=False)

                    for n in range(n_times):
                        working_folder = datetime.now().strftime(f'{ds_folder}/{n}_%Y%m%d_%H.%M')
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
                            retain_points = 1000
                        else:
                            retain_points = X_train.shape[0]
                        pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)

                        dist_train_new = torch.tensor(pairwise_distances(X_train, X_train[pts]), dtype=float32).to(
                            device) / max_val
                        val_dist = torch.tensor(pairwise_distances(X_validation, X_train[pts]), dtype=float32).to(
                            device) / max_val
                        test_dist = torch.tensor(
                            pairwise_distances(X_test, X_train[pts]), dtype=float32).to(device) / max_val

                        X_train = torch.tensor(X_train, dtype=float32).to(device)
                        X_test = torch.tensor(X_test, dtype=float32).to(device)
                        X_validation = torch.tensor(X_validation, dtype=float32).to(device)
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

                        # __________________TRAIN MODEL AS IS__________________________
                        task_model = binary_fnn(X_train.shape[-1]).to(device)
                        task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                        task_criterion = nn.CrossEntropyLoss()
                        task_epochs = 300

                        task_losses = []
                        for ep in range(task_epochs):
                            task_optim.zero_grad()
                            out = task_model(X_train)
                            task_loss = task_criterion(out, y_train.reshape_as(out))
                            # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                            task_loss.backward()
                            task_optim.step()
                            task_losses.append(task_loss.item())

                        train_loss = task_losses[-1]
                        train_accuracy = accuracy(out[:, 0], y_train)

                        out = task_model(X_test)
                        test_loss = task_criterion(out, y_test.reshape_as(out))
                        test_accuracy = accuracy(out[:, 0], y_test)

                        out = task_model(X_validation)
                        val_loss = task_criterion(out, y_validation.reshape_as(out))
                        val_accuracy = accuracy(out[:, 0], y_validation)

                        df = pd.DataFrame()
                        df['CrossEntropy_train'] = [train_loss]
                        df['CrossEntropy_validation'] = [val_loss.cpu().detach().numpy()]
                        df['CrossEntropy_test'] = [test_loss.cpu().detach().numpy()]
                        df['accuracy_train'] = [train_accuracy.cpu().detach().numpy()]
                        df['accuracy_validation'] = [val_accuracy.cpu().detach().numpy()]
                        df['accuracy_test'] = [test_accuracy.cpu().detach().numpy()]
                        df.to_csv(f'{working_folder}/raw_model_metrics.csv', index=False)


                        # ________________TRAIN ISOMAP ______________________________

                        isomap_model = IsomapNN(reduced_dist, n_components=latent_len)
                        isomap_model.to(device)

                        isomap_epochs = 500
                        task_epochs = 300
                        isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.001)
                        isomap_criterion = nn.CrossEntropyLoss()
                        validation_criterion = nn.CrossEntropyLoss()
                        losses = []
                        val_losses = []
                        best_loss = np.inf
                        best_val_loss = np.inf
                        best_isomap_model = None

                        # ISOMAP TRAIN LOOP
                        for epoch in range(isomap_epochs):
                            try:
                                reproj_features = isomap_model().to(float32)
                            except Exception as e:
                                print(e)

                            '''plt.scatter(reproj_features.cpu().detach().numpy()[:, 1],
                                        reproj_features.cpu().detach().numpy()[:, 0],
                                        c=reduced_train_target.cpu().detach().numpy())
                            plt.show()'''

                            with torch.no_grad():
                                features = isomap_model.transform(dist_train_new)

                            task_model = binary_fnn(latent_len).to(device)
                            task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                            task_criterion = nn.CrossEntropyLoss()

                            task_losses = []
                            for ep in range(task_epochs):
                                task_optim.zero_grad()
                                out = task_model(features)
                                task_loss = task_criterion(out, y_train.reshape_as(out))
                                #print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                                task_loss.backward()
                                task_optim.step()
                                task_losses.append(task_loss.item())

                            output = task_model(reproj_features)
                            isomap_loss = isomap_criterion(output.to(torch.float32),
                                                           reduced_train_target.reshape_as(output).to(torch.float32))
                            losses.append(isomap_loss.item())

                            # VALIDATION
                            reproj_val_features = isomap_model.transform(val_dist)
                            val_output = task_model(reproj_val_features)
                            val_loss = validation_criterion(val_output, y_validation.reshape_as(val_output))
                            val_losses.append(val_loss.item())

                            if losses[-1] < best_loss:
                                best_loss = losses[-1]
                                best_isomap_model = isomap_model

                            if val_losses[-1] < best_val_loss:
                                best_val_loss = val_losses[-1]
                                if len(dataset_df) > 250:
                                    best_isomap_model = isomap_model
                                # SAVE OPTIMIZATION PROCESS
                                '''np.save(f'{working_folder}/optimization/distance_matrix_{epoch}.npy',
                                        isomap_model.distances_matrix.cpu().detach().numpy())'''

                            isomap_loss.backward()
                            isomap_optim.step()
                            print(f'{dataset_name} - epoch {epoch}/{isomap_epochs},  loss={losses[-1]}, val_loss={val_losses[-1]}')

                            # ______________SAVE ISOMAP ON EUQLID DIST_______________________
                            if epoch==0:
                                reproj_test_features = isomap_model.transform(test_dist)
                                test_out = task_model(reproj_test_features)
                                test_loss = task_criterion(test_out, y_test.reshape_as(test_out))

                                train_accuracy = accuracy(out[:, 0], y_train)
                                val_accuracy = accuracy(val_output[:, 0], y_validation)
                                test_accuracy = accuracy(test_out[:, 0], y_test)

                                df = pd.DataFrame()
                                df['CrossEntropy_train'] = [losses[-1]]
                                df['CrossEntropy_validation'] = [val_losses[-1]]
                                df['CrossEntropy_test'] = [test_loss.cpu().detach().numpy()]
                                df['accuracy_train'] = [train_accuracy.cpu().detach().numpy()]
                                df['accuracy_validation'] = [val_accuracy.cpu().detach().numpy()]
                                df['accuracy_test'] = [test_accuracy.cpu().detach().numpy()]
                                df.to_csv(f'{working_folder}/euql_isomap_metrics.csv', index=False)


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

                        test_proj_points = best_isomap_model.transform(test_dist)

                        task_model = binary_fnn(latent_len).to(device)
                        task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                        task_criterion = nn.CrossEntropyLoss()
                        for ep in range(task_epochs):
                            task_optim.zero_grad()
                            out = task_model(features)
                            task_loss = task_criterion(out, y_train.reshape_as(out))
                            # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                            task_loss.backward()
                            task_optim.step()

                        train_output = task_model(train_reproj_points.to(float32))
                        val_output = task_model(val_proj_points.to(float32))
                        test_output = task_model(test_proj_points.to(float32))

                        # METRICS CALCULATION
                        score = nn.CrossEntropyLoss()
                        train_score = score(train_output, y_train.reshape_as(train_output)).item()
                        val_score = score(val_output, y_validation.reshape_as(val_output)).item()
                        test_score = score(test_output, y_test.reshape_as(test_output)).item()

                        train_accuracy = accuracy(train_output[:, 0], y_train).cpu().detach().numpy()
                        val_accuracy = accuracy(val_output[:, 0], y_validation).cpu().detach().numpy()
                        test_accuracy = accuracy(test_output[:, 0], y_test).cpu().detach().numpy()

                        df = pd.DataFrame()
                        df['CrossEntropy_train'] = [train_score]
                        df['CrossEntropy_validation'] = [val_score]
                        df['CrossEntropy_test'] = [test_score]
                        df['accuracy_train'] = [train_accuracy]
                        df['accuracy_validation'] = [val_accuracy]
                        df['accuracy_test'] = [test_accuracy]
                        df.to_csv(f'{working_folder}/metrics.csv', index=False)

                        train_proj_points = train_reproj_points.cpu().detach().numpy()
                        val_proj_points = val_proj_points.cpu().detach().numpy()
                        test_proj_points = test_proj_points.cpu().detach().numpy()

                        train_proj_points_2d = PCA(n_components=2).fit_transform(train_proj_points)
                        train_points_2d = PCA(n_components=2).fit_transform(X_train.cpu().detach().numpy())
                        val_proj_points_2d = PCA(n_components=2).fit_transform(val_proj_points)
                        val_points_2d = PCA(n_components=2).fit_transform(X_validation.cpu().detach().numpy())
                        test_proj_points_2d = PCA(n_components=2).fit_transform(test_proj_points)
                        test_points_2d = PCA(n_components=2).fit_transform(X_test.cpu().detach().numpy())

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
                        fig.suptitle(f'Train set CrossEntropy={train_score}, accuracy={train_accuracy}')
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
                        fig.suptitle(f'Validation set CrossEntropy={val_score}, accuracy={val_accuracy}')
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
                        fig.suptitle(f'Test set CrossEntropy={test_score}, accuracy={test_accuracy}')
                        plt.tight_layout()
                        plt.savefig(f'{working_folder}/test_inference.png')
                        plt.close()

                        torch.cuda.empty_cache()

                except KeyError as e:
                    print(f'Skip {dataset_name}\n{e}')

            except Exception as e:
                print(f'{dataset_name} - {e}')
                try:
                    os.rename(ds_folder, f'ICML_RESULTS/broken_{dataset_name}')
                except Exception as e:
                    pass
                pass


run_openml_binary(5)
