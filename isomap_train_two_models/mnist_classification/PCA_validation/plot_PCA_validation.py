import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('PCA_convergence.csv')
for col in df.columns:
    plt.plot(np.arange(len(df[col])), df[col], label=f'n_components={col}')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('CrossEntropyLoss')
plt.title('MNIST classification with PCA')
plt.axhline(df.to_numpy().min(), c='r', linestyle='dashed')
plt.annotate(str(round(df.to_numpy().min(), 4)), (0, df.to_numpy().min()), c='r')
plt.tight_layout()
plt.savefig('PCA_convergence_plot.png', dpi=600)
plt.show()