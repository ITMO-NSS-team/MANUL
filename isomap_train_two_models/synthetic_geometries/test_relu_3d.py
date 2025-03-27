import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, float32
import plotly.graph_objects as go
import plotly.express as px

from isomap_train_two_models.synthetic_geometries.data_generation import torus

device = 'cuda'


def plot_3d_html(x, y, z, colors, name):
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df['colors'] = colors
    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='colors')
    fig.update_scenes(aspectmode='data')
    fig.show()
    fig.write_html(f"{name}.html")




data, colors = torus()
plot_3d_html(data[:, 1], data[:, 0], data[:, 2], colors, 'real')
data = torch.tensor(data, dtype=float32).to(device)
colors = torch.tensor(colors, dtype=float32).to(device)

task_model = nn.Sequential(nn.Linear(data.shape[1], 3, dtype=float32),
                           #nn.ReLU(),
                                           nn.Linear(3, 256, dtype=float32),
                                           nn.Linear(256, 64, dtype=float32),
                                           nn.Linear(64, 1, dtype=float32),
                                           ).to(device)

task_optim = torch.optim.AdamW(params=task_model.parameters(), lr=0.0001)
task_criterion = nn.L1Loss()

task_losses = []
epochs = 1000
for ep in range(epochs):
    task_optim.zero_grad()
    out = task_model(data)
    task_loss = task_criterion(out, colors.reshape_as(out))
    print(f'Task model: epoch {ep}/{epochs}, loss={task_loss.item()}')
    task_losses.append(task_loss.item())
    task_loss.backward()
    task_optim.step()

plt.plot(np.arange(len(task_losses)), task_losses)
plt.show()

predicted = task_model(data).cpu().detach().numpy()
data = data.cpu().detach().numpy()

plot_3d_html(data[:, 1], data[:, 0], data[:, 2], predicted, 'linear_pred')