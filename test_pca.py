# # # PCA # # #

from sklearn.decomposition import PCA

import numpy as np
from sklearn import metrics

def pca_model(X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test):

    num_params = X_train_scaled.shape[1] # number of grid search parameters - ask about this number

    components = np.linspace(1,num_params,num_params,dtype = 'int')
    gridsearch = np.zeros((num_params,2))

    run = 0

    for comp in components:
        model = PCA(comp)
        model.fit_transform(X_train_scaled)
        X_val_pca = model.transform(X_val_scaled)

        X_val_reconstructed = model.inverse_transform(X_val_pca)

        reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed)**2, axis=1) # calculating validation scores through reconstruction errors
        # adjusting y_val to match X_val transformed to a sliding window by cutting off the points lost from the beginning of the array
        y_val = y_val[len(y_val)-len(reconstruction_error):] 
        auc = metrics.roc_auc_score(y_val,reconstruction_error)

        gridsearch[run,:]= np.asarray([comp,auc])

        run +=1

    best_index = np.argmax(gridsearch[:,1])
    highest_val_auc = np.max(gridsearch[:,1])
    best_comp = gridsearch[best_index,0]

    print(f"Best number of components is {best_comp} (AUC score={highest_val_auc})")

    model = PCA(int(best_comp))
    model.fit_transform(X_train_scaled)
    X_test_pca = model.transform(X_test_scaled)

    X_test_reconstructed = model.inverse_transform(X_test_pca)

    reconstruction_error = np.mean((X_test_scaled - X_test_reconstructed)**2, axis=1)
    auc = metrics.roc_auc_score(y_test[len(y_test)-len(reconstruction_error):],reconstruction_error)

    print("Final AUC is", auc)

    # parameter scatterplot parameters

    x = gridsearch[:,0]
    y = gridsearch[:,1]

    return x, y, reconstruction_error, auc








