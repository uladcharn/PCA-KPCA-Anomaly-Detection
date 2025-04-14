# # # PCA # # #

from preprocessing import loading_sets_cv, scaling_and_sliding_window_reformating_X

from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
from sklearn import metrics
import pandas as pd

def pca_model(data): # , X_train_scaled, X_val_scaled, X_test_scaled, y_val, y_test

    num_params = 40

    components = np.linspace(1,num_params,num_params,dtype = 'int')
    gridsearch = np.zeros((num_params,2))

    run = 0

    train_val_data, test_data = loading_sets_cv(data)

    normal_data = train_val_data[train_val_data['anomaly'] == 0]
    anomaly_data = train_val_data[train_val_data['anomaly'] == 1] 

    X_test, y_test = test_data.drop(columns=["anomaly"]).values, test_data["anomaly"].values
    
    tscv = TimeSeriesSplit(n_splits=5) 
    
    for comp in components:
        #implement your cross validation here
        auc_scores = []        

        for train_idx, val_idx in tscv.split(normal_data):
            train_data = normal_data.iloc[train_idx]
            val_normal = normal_data.iloc[val_idx]
            
            val_anomaly = anomaly_data[anomaly_data.index.isin(val_normal.index)]
            if val_anomaly.empty:
                val_anomaly = anomaly_data.sample(3, random_state=44)
            val_data = pd.concat([val_normal, val_anomaly]).sort_index()

            X_train, X_val = train_data.drop(columns=["anomaly"]).values, val_data.drop(columns=["anomaly"]).values
            y_val = val_data["anomaly"].values

            X_train_scaled, X_val_scaled, _ = scaling_and_sliding_window_reformating_X(X_train, X_val, None, scale_test=False)

            model = PCA(comp)
            model.fit_transform(X_train_scaled)
            X_val_pca = model.transform(X_val_scaled)

            X_val_reconstructed = model.inverse_transform(X_val_pca)

            reconstruction_error = np.mean((X_val_scaled - X_val_reconstructed)**2, axis=1) # calculating validation scores through reconstruction errors
            # adjusting y_val to match X_val transformed to a sliding window by cutting off the points lost from the beginning of the array
            y_val = y_val[len(y_val)-len(reconstruction_error):] 
            auc = metrics.roc_auc_score(y_val,reconstruction_error)
            auc_scores.append(auc)

        avg_auc = np.mean(auc_scores)

        gridsearch[run,:]= np.asarray([comp,avg_auc])

        run +=1

    best_index = np.argmax(gridsearch[:,1])
    highest_val_auc = np.max(gridsearch[:,1])
    best_comp = gridsearch[best_index,0]

    print(f"Best number of components is {best_comp} (AUC score={highest_val_auc})")

    model = PCA(int(best_comp))
    model.fit_transform(X_train_scaled)

    _, _, X_test_scaled = scaling_and_sliding_window_reformating_X(X_test, X_val, X_test)

    X_test_pca = model.transform(X_test_scaled)

    X_test_reconstructed = model.inverse_transform(X_test_pca)

    reconstruction_error = np.mean((X_test_scaled - X_test_reconstructed)**2, axis=1)
    test_auc = metrics.roc_auc_score(y_test[len(y_test)-len(reconstruction_error):],reconstruction_error)

    print("Final AUC is", test_auc)

    # parameter scatterplot parameters

    x = gridsearch[:,0]
    y = gridsearch[:,1]

    return x, y, reconstruction_error, test_auc








