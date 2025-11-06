import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

H = np.load('sionna_sample.npy')  # [64, 4, 256, 20, 4]
H_flat = H.reshape(64, 4, 256, -1)  # [64, 4, 256, 80]

amplitude = np.abs(H_flat)        # [64, 4, 256, 80]
phase = np.angle(H_flat)          # [64, 4, 256, 80]

H_processed = np.stack([amplitude, phase], axis=-1)  # [64, 4, 256, 80, 2]
H_processed = H_processed.reshape(64 * 4 * 256 * 2, 80).T  # [80, 131072]

# Нормализация
scaler = StandardScaler()
H_normalized = scaler.fit_transform(H_processed)

# PCA
pca_2d = PCA(n_components=2)
pca_3d = PCA(n_components=3)

H_pca_2d = pca_2d.fit_transform(H_normalized)
H_pca_3d = pca_3d.fit_transform(H_normalized)

# Explained variance
explained_2d = np.sum(pca_2d.explained_variance_ratio_)
explained_3d = np.sum(pca_3d.explained_variance_ratio_)

# 2D Plot with point numbers
plt.figure(figsize=(8, 6))
plt.scatter(H_pca_2d[:, 0], H_pca_2d[:, 1], alpha=0.7)

# Add point numbers
for i in range(len(H_pca_2d)):
    plt.annotate(str(i), (H_pca_2d[i, 0], H_pca_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'2D PCA (Explained Variance: {explained_2d:.3f})')
plt.grid(True, alpha=0.3)
plt.show()

# 3D Plot with point numbers
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(H_pca_3d[:, 0], H_pca_3d[:, 1], H_pca_3d[:, 2], alpha=0.7)

# Add point numbers
for i in range(len(H_pca_3d)):
    ax.text(H_pca_3d[i, 0], H_pca_3d[i, 1], H_pca_3d[i, 2],
            str(i), fontsize=8)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title(f'3D PCA (Explained Variance: {explained_3d:.3f})')
plt.tight_layout()
plt.show()

print(f"2D Explained Variance: {explained_2d:.4f}")
print(f"3D Explained Variance: {explained_3d:.4f}")