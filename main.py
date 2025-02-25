import matplotlib.pyplot as plt
import numpy as np
import sys

from preprocessing import preprocess_data, loading_sets, scaling_and_sliding_window_reformating_X
from test_kpca import kpca_model
from test_pca import pca_model

from sklearn import metrics

np.random.seed(44)

data_dict = {
    "datasets":["circuit_water.csv", "fluid_leaks.csv", "rotor_imbalance.csv"],
    "labels": ["Circuit Water", "Fluid Leaks", "Rotor Imbalance"],
    "number": ["1","2","3"]
    }

num = sys.argv[1]

assert num in data_dict["number"], "Invalid dataset number entered."

index = data_dict["number"].index(num)

filepath = data_dict["datasets"][index]

data = preprocess_data(filepath)
print("The number of anomalies in a dataset is ", sum(data['anomaly']))

X_train, X_val, y_val, X_test, y_test = loading_sets(data)

X_train_scaled, X_val_scaled, X_test_scaled = scaling_and_sliding_window_reformating_X(X_train, X_val, X_test)

error_scores = []
auc_scores = []
methods = ['PCA','KernelPCA']

fig, ax = plt.subplots(1,2,figsize=(10,5))

## PCA ## 

x_pca, y_pca, error_scores_pca, auc_pca = pca_model(X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test)
error_scores.append(error_scores_pca)
auc_scores.append(auc_pca)

c = ax[0].scatter(x_pca, y_pca)
ax[0].set_title('PCA')

ax[0].axis([x_pca.min(),x_pca.max(),y_pca.min(),y_pca.max()])

ax[0].set_xlabel('n_comps')
ax[0].set_ylabel('AUC')

## KernelPCA ## 

x_kpca, y_kpca, z, best_gamma, best_comp, error_scores_kpca, auc_kpca = kpca_model(X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test) 
error_scores.append(error_scores_kpca)
auc_scores.append(auc_kpca)

c = ax[1].pcolormesh(x_kpca, y_kpca, z,  cmap='viridis', vmin=z.min(), vmax=z.max()) 
ax[1].set_title('Kernel PCA')
ax[1].axis([x_kpca.min(),x_kpca.max(),y_kpca.min(),y_kpca.max()])
ax[1].set_xscale('log')
fig.colorbar(c, ticks=[z.min(), np.median(z),z.max()]) 
ax[1].scatter(best_gamma, best_comp, marker = "*", s = 10, color = 'r')
ax[1].set_ylabel('n_components')
ax[1].set_xlabel('gamma')

plt.subplots_adjust(hspace=0.5, wspace=0.5)
fig.suptitle(data_dict["labels"][index], fontsize="x-large")

fig.savefig(f"full_plot_{num}(2)")

# Making ROC Curves

fig, ax = plt.subplots(figsize=(4,4))
colors = ['g','r']

for i in range(len(methods)):
    false_pos_r, true_pos_r, _ = metrics.roc_curve(y_test[len(y_test)-len(error_scores[i]):], error_scores[i], drop_intermediate=False)
    true_pos_r[1] = 0
    if auc_scores[i] == 0.5:
        false_pos_r = np.linspace(0,1,100)
        true_pos_r = false_pos_r
    ax.plot(false_pos_r, true_pos_r, color = colors[i], marker='o', label=methods[i], ms=2)

x_roc = y_roc = np.linspace(0,1,100)
ax.plot(x_roc,y_roc, ls="--", label='random')
ax.set_xlim([1e-4,1])
ax.legend()
ax.set_title(data_dict["labels"][index] + ' - ROC')
ax.set_xlabel('False Positive Rate', fontsize = 'medium')
ax.set_ylabel('True Positive Rate', fontsize = 'medium')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(f"roc_{num}(2)")


