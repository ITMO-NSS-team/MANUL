import torch
import torch.nn as nn
import torch.optim as optim

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 1, 28, 28]
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent




# Load MNIST dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=False)
mnist_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

# Initialize model, loss, and optimizer
latent_dim = 32
model = ConvAutoencoder(latent_dim=latent_dim).to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for batch, _ in mnist_loader:
        batch = batch.to("cuda")
        optimizer.zero_grad()
        reconstruction, _ = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(mnist_loader)}")


from sklearn.manifold import Isomap

# Extract latent representations
all_latent = []
for batch, _ in mnist_loader:
    batch = batch.to("cuda")
    _, latent = model(batch)
    all_latent.append(latent.detach().cpu())
all_latent = torch.cat(all_latent, dim=0)

# Perform Isomap
isomap = Isomap(n_components=latent_dim, n_neighbors=5)
isomap_embedding = isomap.fit_transform(all_latent.numpy())


# Map Isomap embedding back to the latent space
isomap_embedding = torch.tensor(isomap_embedding, dtype=torch.float32).to("cuda")

# Decode Isomap embedding into images
reconstructed_images = model.decoder(isomap_embedding).detach().cpu()

# Visualize reconstructed images
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i, ax in enumerate(axes):
    ax.imshow(reconstructed_images[i].squeeze(0), cmap="gray")
    ax.axis("off")
plt.show()