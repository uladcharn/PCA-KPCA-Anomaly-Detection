import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    
    data = pd.read_csv(filepath, header=0, sep=';')

    # Exclude datetime column
    data_values = data.drop(columns=['datetime','changepoint'], axis = 1)

    # Convert data to float type
    data_values = data_values.astype('float32')

    # Drop missing values
    data_values = data_values.dropna()

    return data_values

def loading_sets(data, train_ratio = 0.6): # time series requires chronological splitting rather than random sampling 
    normal_data = data[data['anomaly'] == 0]
    anomaly_data = data[data['anomaly'] == 1] 

    normal_data_X = normal_data.drop(columns=["anomaly"])
    normal_data_y = normal_data["anomaly"]

    anomaly_data_X = anomaly_data.drop(columns=["anomaly"])
    anomaly_data_y = anomaly_data["anomaly"]

    normal_train_X, normal_temp_X = train_test_split(normal_data_X, test_size=1-train_ratio, shuffle=False)
    _, normal_temp_y = train_test_split(normal_data_y, test_size=1-train_ratio, shuffle=False)
    normal_val_X, normal_test_X = train_test_split(normal_temp_X, test_size=0.5, shuffle=False) 
    normal_val_y, normal_test_y = train_test_split(normal_temp_y, test_size=0.5, shuffle=False)

    anomaly_train_X, anomaly_temp_X = train_test_split(anomaly_data_X, test_size=1-train_ratio, shuffle=False)
    _, anomaly_temp_y = train_test_split(anomaly_data_y, test_size=1-train_ratio, shuffle=False)
    anomaly_val_X, anomaly_test_X = train_test_split(anomaly_temp_X, test_size=0.5, shuffle=False)
    anomaly_val_y, anomaly_test_y = train_test_split(anomaly_temp_y, test_size=0.5, shuffle=False)

    X_train = np.concatenate((normal_train_X,anomaly_train_X), axis=0)
    X_val, y_val = np.concatenate((normal_val_X,anomaly_val_X), axis=0), np.concatenate((normal_val_y,anomaly_val_y), axis=0)
    X_test, y_test = np.concatenate((normal_test_X,anomaly_test_X), axis=0), np.concatenate((normal_test_y,anomaly_test_y), axis=0)

    return X_train, X_val, y_val, X_test, y_test 

def create_sliding_window(data, window_size=20):
    return np.array([data[i: i + window_size] for i in range(len(data) - window_size)])

def scaling_and_sliding_window_reformating_X(X_train, X_val, X_test, window_size = 20):
    X_train = create_sliding_window(X_train,window_size)
    X_val = create_sliding_window(X_val,window_size)
    X_test = create_sliding_window(X_test,window_size)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(len(X_train), -1))
    X_val_scaled = scaler.transform(X_val.reshape(len(X_val), -1))
    X_test_scaled = scaler.transform(X_test.reshape(len(X_test), -1))

    return X_train_scaled, X_val_scaled, X_test_scaled