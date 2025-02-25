# # # KernelPCA # # #

from sklearn.decomposition import KernelPCA

import numpy as np
from sklearn import metrics

def kpca_model(X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test):

    num_params = X_train_scaled.shape[1]
    gammas = np.logspace(-2,2,num_params)

    max_num_comps = X_train_scaled.shape[1]

    components = np.linspace(1,max_num_comps,num_params, dtype='int')

    gridsearch = np.zeros((num_params*num_params,3))
    mesh = np.zeros((num_params,num_params)) # for parameter heatmaps

    run = 0
    i = 0

    for gamma in gammas:
        j = 0
        print(gamma)
        for comp in components:
            model = KernelPCA(comp, gamma=gamma, fit_inverse_transform=True)
            model.fit_transform(X_train_scaled)
            X_val_kpca = model.transform(X_val_scaled)

            X_val_reconstructed = model.inverse_transform(X_val_kpca)
            # reconstruction_error = np.linalg.norm(X_val_scaled - X_val_reconstructed, axis=1)
            reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed)**2, axis=1)
            y_val = y_val[len(y_val)-len(reconstruction_error):] # to adapt y_val to the window-formatted x's 

            auc = metrics.roc_auc_score(y_val,reconstruction_error)
            gridsearch[run,:]= np.asarray([gamma,comp,auc])
            mesh[j,i]=auc

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
    X_test_kpca = model.transform(X_test_scaled)

    X_test_reconstructed = model.inverse_transform(X_test_kpca)

    # reconstruction_error = np.linalg.norm(X_test_scaled - X_test_reconstructed, axis=1) 
    reconstruction_error = np.mean((X_test_scaled - X_test_reconstructed)**2, axis=1)
    auc = metrics.roc_auc_score(y_test[len(y_test)-len(reconstruction_error):],reconstruction_error) 

    print("Final AUC is", auc)

    # parameter heatmap parameters

    x, y = np.meshgrid(gammas, components)

    z = mesh

    return x, y, z, best_gamma, best_comp, reconstruction_error, auc

