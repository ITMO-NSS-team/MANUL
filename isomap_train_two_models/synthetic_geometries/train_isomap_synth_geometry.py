import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from torch import float32, nn

from isomap_train_two_models.Isomap import IsomapNN
from isomap_train_two_models.utils import reduce_dist_fps
from sklearn.model_selection import train_test_split

from isomap_train_two_models.synthetic_geometries.data_generation import geometries
import plotly.express as px

device = 'cuda'


def plot_3d_html(points, colors, path_to_save):
    df = pd.DataFrame()
    df['x'] = points[:, 0]
    df['y'] = points[:, 1]
    df['z'] = points[:, 2]
    df['colors'] = colors
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='colors')
    fig.update_scenes(aspectmode='data')
    fig.write_html(path_to_save)


def plot_predictoion_PCA_transform(points, proj_points, true_labels, predicted_labels, title, save_path):
    points_2d = PCA(n_components=2).fit_transform(points)
    proj_points_2d = PCA(n_components=2).fit_transform(proj_points)

    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    cs0 = axs[0].scatter(points_2d[:, 1], points_2d[:, 0], c=true_labels)
    fig.colorbar(cs0, ax=axs[0])
    axs[0].set_title('Euclidean - Target values')

    cs1 = axs[1].scatter(proj_points_2d[:, 1], proj_points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs1, ax=axs[1])
    axs[1].set_title('ISOMAP projected - Predicted values')

    cs2 = axs[2].scatter(points_2d[:, 1], points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs2, ax=axs[2])
    axs[2].set_title('Euclidean - Predicted values')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictoion(points, proj_points_2d, true_labels, predicted_labels, title, save_path):
    points_2d = PCA(n_components=2).fit_transform(points)

    fig, axs = plt.subplots(2, 2, figsize=(14, 5))
    cs0 = axs[0,0].scatter(points_2d[:, 1], points_2d[:, 0], c=true_labels)
    fig.colorbar(cs0, ax=axs[0,0])
    axs[0,0].set_title('Euclidean - Target values')

    cs1 = axs[1,0].scatter(proj_points_2d[:, 1], proj_points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs1, ax=axs[1,0])
    axs[1,0].set_title('ISOMAP projected - Predicted values')

    cs3 = axs[1,1].scatter(proj_points_2d[:, 1], proj_points_2d[:, 0], c=true_labels)
    fig.colorbar(cs1, ax=axs[1,1])
    axs[1,1].set_title('ISOMAP projected - True values')

    cs2 = axs[0,1].scatter(points_2d[:, 1], points_2d[:, 0], c=predicted_labels)
    fig.colorbar(cs2, ax=axs[0,1])
    axs[0,1].set_title('Euclidean - Predicted values')

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_3d_prediction(points, proj_points, true_labels, predicted_labels, title, save_path):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')

    ax1.scatter(points[:, 1], points[:, 0], points[:, 2], c=true_labels)
    ax1.set_title('Target')
    ax2.scatter(proj_points[:, 1], proj_points[:, 0], proj_points[:, 2], c=predicted_labels)
    ax2.set_title('Transformed geometry')
    ax3.scatter(points[:, 1], points[:, 0], points[:, 2], c=predicted_labels)
    ax3.set_title('Prediction')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_isomap(run_number=1):
    for geometry in ['hyperboloid_of_one_sheet', 'pseudosphere']:
        #for geometry in geometries.keys():
        data, labels = geometries[geometry]()
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        X_train, X_test, y_train, y_test = train_test_split(data, labels)

        print('X_train mean={} std={}'.format(np.mean(X_train),np.var(X_train)))
        print('X_test mean={} std={}'.format(np.mean(X_test),np.var(X_test)))

        geom_path = f'results_(1k)/{geometry}/isomap'
        if not os.path.exists(geom_path):
            os.makedirs(geom_path)

        dist_train = torch.tensor(pairwise_distances(X_train, X_train), dtype=float32)
        max_val = torch.max(dist_train)

        
        dist_train = dist_train / max_val
        # SELECT SPARSE POINTS
        if X_train.shape[0] > 5000:
            retain_points = 1000
            pts, reduced_dist = reduce_dist_fps(dist_train, retain_points)
        else:
            reduced_dist = torch.tensor(dist_train, dtype=float32).to(device)
            pts = np.arange(X_train.shape[0])

        dist_train_new = torch.tensor(pairwise_distances(X_train, X_train[pts]), dtype=float32).to(
            device) / max_val
        test_dist = torch.tensor(
            pairwise_distances(X_test, X_train[pts]), dtype=float32).to(device) / max_val

        train_target = torch.tensor(y_train, dtype=float32).to(device)
        test_target = torch.tensor(y_test, dtype=float32).to(device)
        reduced_train_target = torch.tensor(y_train[pts], dtype=float32).to(device)

        # reduce features into isomap is its extensive
        '''if X_train.shape[-1] > 15:
            latent_len = 15
        else:
            latent_len = X_train.shape[-1]'''
        latent_len = 3
        print('reduced_dist mean={} std={}'.format(torch.mean(reduced_dist),torch.var(reduced_dist)))
        print('dist_train_new mean={} std={}'.format(torch.mean(dist_train_new),torch.var(dist_train_new)))
        print('test_dist mean={} std={}'.format(torch.mean(test_dist),torch.var(test_dist)))
        for r in range(0, run_number):
            print(f'{geom_path} - {r}')

            # ________________TRAIN ISOMAP ______________________________
            
            isomap_model = IsomapNN(reduced_dist, n_components=latent_len, eigval_choice='MDS')
            isomap_model.to(device)

            isomap_epochs = 1000
            task_epochs = 150
            isomap_optim = torch.optim.AdamW(params=isomap_model.parameters(), lr=0.001)
            isomap_criterion = nn.L1Loss()
            addit_criterion = nn.MSELoss()
            losses = []
            best_loss = np.inf
            best_isomap_model = None

            # ISOMAP TRAIN LOOP
            for epoch in range(isomap_epochs):
                reproj_features = isomap_model().to(float32)
                #print('reproj_features mean={} std={}'.format(torch.mean(reproj_features),torch.var(reproj_features)))
                with torch.no_grad():
                    
                    features = isomap_model.transform(dist_train_new)
                    #print('features mean={} std={}'.format(torch.mean(features),torch.var(features)))
                task_model = nn.Sequential(nn.Linear(latent_len, 512, dtype=float32),
                                           #nn.ReLU(),
                                           nn.Linear(512, 256, dtype=float32),
                                           nn.Linear(256, 64, dtype=float32),
                                           nn.Linear(64, 1, dtype=float32)
                                           ).to(device)
                task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
                task_criterion = nn.L1Loss()

                task_losses = []
                for ep in range(task_epochs):
                    task_optim.zero_grad()
                    out = task_model(features)
                    task_loss = task_criterion(out, train_target.reshape_as(out))
                    # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                    task_losses.append(task_loss.item())
                    task_loss.backward()
                    task_optim.step()

                output = task_model(reproj_features)
                isomap_loss = isomap_criterion(output.to(torch.float32),
                                               reduced_train_target.reshape_as(output).to(torch.float32))
                losses.append(isomap_loss.item())

                if losses[-1] < best_loss:
                    best_loss = losses[-1]
                    best_isomap_model = isomap_model

                if losses[-1] > 0.20:
                    for g in isomap_optim.param_groups:
                        g['lr'] = 0.001
                if 0.20 >= losses[-1] >= 0.06:
                    for g in isomap_optim.param_groups:
                        g['lr'] = 0.0001
                if losses[-1] <= 0.06:
                    for g in isomap_optim.param_groups:
                        g['lr'] = 0.00001



                '''clip_value = 2
                for p in isomap_model.parameters():
                    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))'''
                if epoch % 10 == 0:
                    isomap_loss.backward()
                    isomap_optim.step()
                print(f'{geometry} - epoch {epoch}/{isomap_epochs},  loss={losses[-1]}')

                # ______________SAVE ISOMAP ON EUQLID DIST_______________________
                if epoch == 0:
                    reproj_test_features = isomap_model.transform(test_dist)
                    test_out = task_model(reproj_test_features)
                    test_loss = task_criterion(test_out.reshape_as(test_target), test_target)

                    plot_3d_html(points=X_train,
                                 colors=out.cpu().detach().numpy(),
                                 path_to_save=f'{geom_path}/{r}_isomap_raw_train.html')

                    train_mae = addit_criterion(out.reshape_as(train_target), train_target)
                    test_mae = addit_criterion(test_out.reshape_as(test_target), test_target)

                    error_metric = pd.DataFrame()
                    error_metric['train_mae'] = [losses[-1]]
                    error_metric['train_mse'] = [train_mae.item()]
                    error_metric['test_mae'] = [test_loss.item()]
                    error_metric['test_mse'] = [test_mae.item()]
                    error_metric.to_csv(f'{geom_path}/{r}_isomap_raw_metrics.csv', index=False)

            plt.figure()
            plt.plot(np.arange(len(losses)), losses, label='Train')
            plt.axhline(best_loss, c='r', linestyle='dashed')
            plt.annotate(str(round(best_loss, 4)), (0, best_loss), c='r')
            plt.title('Convergence plot')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend()
            plt.tight_layout()
            plt.yscale('log')
            plt.savefig(f'{geom_path}/{r}_isomap_model_convergence.png')
            # plt.show()
            plt.close()

            # ISOMAP POINTS PROJECTION AFTER OPTIMIZATION
            with torch.no_grad():
                train_reproj_points = best_isomap_model.transform(dist_train_new)
            test_proj_points = best_isomap_model.transform(test_dist)

            task_model = nn.Sequential(nn.Linear(latent_len, 562, dtype=float32),
                                       # nn.ReLU(),
                                       nn.Linear(562, 256, dtype=float32),
                                       nn.Linear(256, 64, dtype=float32),
                                       nn.Linear(64, 1, dtype=float32)
                                       ).to(device)
            task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
            task_criterion = nn.L1Loss()
            for ep in range(task_epochs):
                task_optim.zero_grad()
                out = task_model(train_reproj_points)
                task_loss = task_criterion(out.reshape_as(train_target), train_target)
                # print(f'Task model: epoch {ep}/{task_epochs}, loss={task_loss.item()}')
                task_loss.backward()
                task_optim.step()

            train_output = task_model(train_reproj_points.to(float32))
            test_output = task_model(test_proj_points.to(float32))

            # METRICS CALCULATION
            score = nn.L1Loss()

            train_loss = score(train_output.reshape_as(train_target), train_target)
            test_loss = score(test_output.reshape_as(test_target), test_target)
            train_mae = addit_criterion(train_output.reshape_as(train_target), train_target)
            test_mae = addit_criterion(test_output.reshape_as(test_target), test_target)

            error_metric = pd.DataFrame()
            error_metric['train_mae'] = [train_loss.item()]
            error_metric['train_mse'] = [train_mae.item()]
            error_metric['test_mae'] = [test_loss.item()]
            error_metric['test_mse'] = [test_mae.item()]
            error_metric.to_csv(f'{geom_path}/{r}_isomap_optimized_metrics.csv', index=False)

            train_proj_points = train_reproj_points.cpu().detach().numpy()
            test_proj_points = test_proj_points.cpu().detach().numpy()

            plot_3d_prediction(points=X_train,
                               proj_points=train_proj_points,
                               true_labels=y_train,
                               predicted_labels=train_output.cpu().detach().numpy(),
                               title=f'Train:MSE={train_mae}, MAE={train_loss}',
                               save_path=f'{geom_path}/{r}_isomap_optimized_train_3d.png')
            plot_3d_prediction(points=X_test,
                               proj_points=test_proj_points,
                               true_labels=y_test,
                               predicted_labels=test_output.cpu().detach().numpy(),
                               title=f'Test:MSE={test_mae}, MAE={test_loss}',
                               save_path=f'{geom_path}/{r}_isomap_optimized_test_3d.png')

            plot_predictoion(points=X_train,
                             proj_points_2d=train_proj_points,
                             true_labels=y_train,
                             predicted_labels=train_output.cpu().detach().numpy(),
                             title=f'MSE={train_mae}, MAE={train_loss}',
                             save_path=f'{geom_path}/{r}_isomap_optimized_train.png')
            plot_predictoion(points=X_test,
                             proj_points_2d=test_proj_points,
                             true_labels=y_test,
                             predicted_labels=test_output.cpu().detach().numpy(),
                             title=f'MSE={test_mae}, MAE={test_loss}',
                             save_path=f'{geom_path}/{r}_isomap_optimized_test.png')
            plot_3d_html(points=X_train,
                         colors=train_output.cpu().detach().numpy(),
                         path_to_save=f'{geom_path}/{r}_isomap_optimized_train.html')


run_isomap()
