import os

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from scipy import stats
import pandas as pd

labels = []
pvals = []
errors_df = pd.DataFrame()
variances_df = pd.DataFrame()


for exp in os.listdir('ICML_RESULTS'):
    try:
        df = pd.read_csv(f'ICML_RESULTS/{exp}/accuracy_test.csv')

        pval = stats.friedmanchisquare(df['raw_model'], df['isomap_raw'],
                                       df['isomap_optim'])
        pvals.append(pval[1])

        labels.append(exp)
        mean_df = df.mean()
        var_df = df.std(ddof=0) * 1.96
        errors_df[exp] = mean_df
        variances_df[exp] = var_df
    except Exception as e:
        pass

errors_df.style.apply(lambda col: ['font-weight:bold' if x == col.min() else '' for x in col])
errors_df = errors_df.T
errors_df.columns = df.columns
variances_df = variances_df.T
variances_df.columns = df.columns

print(errors_df.to_string())

errors_df['pvals'] = pvals
variances_df['pvals'] = pvals

labels = errors_df.index.values.tolist()

test_errors_df = errors_df[['raw_model', 'isomap_raw', 'isomap_optim']].to_numpy()
vars_errors_df = variances_df[['raw_model', 'isomap_raw', 'isomap_optim']].to_numpy()
b = np.zeros_like(test_errors_df)

for i in range(b.shape[0]):
    if errors_df['pvals'][i] > 0.05 or np.isnan(errors_df['pvals'][i]):
        b[i, :] = 1
    else:
        if (test_errors_df[i, 2] == np.max(test_errors_df[i]) and
                test_errors_df[i, 0] != np.max(test_errors_df[i])
                and test_errors_df[i, 1] != np.max(test_errors_df[i])):
            b[i, :] = 2
        else:
            b[i, :] = 0

cmap = ListedColormap(["orange", "lightgrey", "palegreen"])
im = plt.imshow(b, aspect="auto", cmap=cmap)

for i in range(test_errors_df.shape[0]):
    for j in range(test_errors_df.shape[1]):
        plt.text(j, i, f'{round(test_errors_df[i, j], 3)}+-{round(vars_errors_df[i, j], 3)}', ha="center",
                 va="center")
plt.yticks(ticks=np.arange(len(labels)), labels=labels)
plt.xticks(ticks=np.arange(3), labels=['Raw model', 'Model+ISOMAP', 'ISOMAP optimized'])
plt.title('OpenML binary datasets - Accuracy')
plt.tight_layout()
#plt.savefig(f'{log_folder}/disp_bin_openml_50-20000.png', dpi=300)
plt.show()