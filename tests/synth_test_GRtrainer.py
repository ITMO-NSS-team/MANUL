import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from regularizator.GraphRegTrainer import GraphRegTrainer
from sklearn.metrics.pairwise import euclidean_distances


X, y = make_regression(n_samples=5000, n_features=10, n_informative=8,
                       noise=10, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

n_basis = 1000
basis_indices = np.random.choice(X_train.shape[0], size=n_basis, replace=False)
basis_indices = np.sort(basis_indices)

X_basis = X_train[basis_indices]

distances = euclidean_distances(X_basis, X_basis)
sigma = np.median(distances)
weights_matrix = np.exp(-distances**2 / (2 * sigma**2))

print(f"Basis points: {n_basis}")
print(f"Weights matrix shape: {weights_matrix.shape}")


trainer = GraphRegTrainer(
    train_features=X_train,
    train_target=y_train,
    weights_matrix=weights_matrix,
    basis_indices=basis_indices,
    model=None,
    num_epochs=500,
    batch_size=32,
    lambda_graph=0.00001,
    n_neighbors=10,
    method='ensemble_knn',
    verbose=True
)

print("\nTraining model...")
trainer.train(plot_convergence=True, adaptive_lambda=False)

print("\nEvaluating on test set")
mse = trainer.evaluate(X_test, y_test)
print(f"Test MSE: {mse:.4f}")

predictions = trainer.predict(X_test)
print(f"\nPredictions shape: {predictions.shape}")
print(f"First 5 predictions: {predictions[:5].flatten()}")
print(f"First 5 true values: {y_test[:5]}")

trainer.save_weights("test_model.pth")
