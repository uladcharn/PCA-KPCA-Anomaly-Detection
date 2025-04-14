# # # KernelPCA # # #

from preprocessing import loading_sets_cv, scaling_and_sliding_window_reformating_X

from sklearn.decomposition import KernelPCA
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
from sklearn import metrics
import pandas as pd

def kpca_model(data): 

    num_params = 20
    gammas = np.logspace(-2,2,num_params)

    max_num_comps = 40

    components = np.linspace(1,max_num_comps,num_params, dtype='int')

    gridsearch = np.zeros((num_params*num_params,3))
    mesh = np.zeros((num_params,num_params)) # for parameter heatmaps

    run = 0
    i = 0

    train_val_data, test_data = loading_sets_cv(data)

    normal_data = train_val_data[train_val_data['anomaly'] == 0]
    anomaly_data = train_val_data[train_val_data['anomaly'] == 1] 

    X_test, y_test = test_data.drop(columns=["anomaly"]).values, test_data["anomaly"].values

    tscv = TimeSeriesSplit(n_splits=5)

    for gamma in gammas:
        j = 0
        print(gamma)
        for comp in components:
            auc_scores = []

            for train_idx, val_idx in tscv.split(normal_data):
                train_data = normal_data.iloc[train_idx]
                val_normal = normal_data.iloc[val_idx]
                
                val_anomaly = anomaly_data[anomaly_data.index.isin(val_normal.index)]
                if val_anomaly.empty: # if the anomaly data is empty, sample some from the general anomaly sub-dataset
                    val_anomaly = anomaly_data.sample(3, random_state=44) 
                val_data = pd.concat([val_normal, val_anomaly]).sort_index()

                X_train, X_val = train_data.drop(columns=["anomaly"]).values, val_data.drop(columns=["anomaly"]).values
                y_val = val_data["anomaly"].values

                X_train_scaled, X_val_scaled, _ = scaling_and_sliding_window_reformating_X(X_train, X_val, None, scale_test=False)

                model = KernelPCA(comp, gamma=gamma, fit_inverse_transform=True)
                model.fit_transform(X_train_scaled)
                X_val_kpca = model.transform(X_val_scaled)

                X_val_reconstructed = model.inverse_transform(X_val_kpca)
                # reconstruction_error = np.linalg.norm(X_val_scaled - X_val_reconstructed, axis=1)
                reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed)**2, axis=1)
                y_val = y_val[len(y_val)-len(reconstruction_error):] # to adapt y_val to the window-formatted x's 

                auc = metrics.roc_auc_score(y_val,reconstruction_error)
                auc_scores.append(auc)
            avg_auc = np.mean(auc_scores)
            gridsearch[run,:]= np.asarray([gamma,comp,avg_auc])
            mesh[j,i]=avg_auc

            run += 1
            j += 1
        i += 1

    best_index = np.argmax(gridsearch[:,2])
    highest_val_auc = np.max(gridsearch[:,2])
    best_gamma = gridsearch[best_index,0]
    best_comp = gridsearch[best_index,1]

    print(f"Best number of components is {best_comp} (AUC score={highest_val_auc}); Best gamma is {best_gamma}.")

    model = KernelPCA(n_components=int(best_comp), kernel="rbf", gamma=best_gamma, fit_inverse_transform=True)
    model.fit_transform(X_train_scaled)

    _, _, X_test_scaled = scaling_and_sliding_window_reformating_X(X_test, X_val, X_test)

    X_test_kpca = model.transform(X_test_scaled)

    X_test_reconstructed = model.inverse_transform(X_test_kpca)

    # reconstruction_error = np.linalg.norm(X_test_scaled - X_test_reconstructed, axis=1) 
    reconstruction_error = np.mean((X_test_scaled - X_test_reconstructed)**2, axis=1)
    test_auc = metrics.roc_auc_score(y_test[len(y_test)-len(reconstruction_error):],reconstruction_error) 

    print("Final AUC is", test_auc)

    # parameter heatmap parameters

    x, y = np.meshgrid(gammas, components)

    z = mesh

    return x, y, z, best_gamma, best_comp, reconstruction_error, test_auc

