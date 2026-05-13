import numpy as np
from utils import mae_mse_chart, eval_chart, price_to_error_chart

W1 = W2 = W3 = W4 = b1 = b2 = b3 = b4 = None

def initialize_layers(H1_size, H2_size, H3_size):
    global W1, W2, W3, W4, b1, b2, b3, b4
    W1 = np.random.randn(9, H1_size ) * np.sqrt(2./9)
    b1 = np.zeros((1, H1_size ))
    W2 = np.random.randn(H1_size , H2_size) * np.sqrt(2./H1_size)
    b2 = np.zeros((1, H2_size))
    W3 = np.random.randn(H2_size, H3_size) * np.sqrt(2./H2_size) 
    b3 = np.zeros((1, H3_size))
    W4 = np.random.randn(H3_size, 1) * np.sqrt(2./H3_size)  
    b4 = np.zeros((1, 1))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def cost(y_hat, y):
    m = y.shape[0]
    return (1 / (2 * m)) * np.sum((y_hat - y)**2)

def predict(ds, X):
    y_norm_pred, _ = feed_forward(X)
    return y_norm_pred * ds.y_std + ds.y_mean

def feed_forward(A0):
    Z1 = A0 @ W1 + b1 
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = relu(Z3)
    Z4 = A3 @ W4 + b4
    cache = {"A0": A0, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
    return Z4, cache

def backprop(y_hat, Y, cache):
    global W1, W2, W3, W4, b1, b2, b3, b4
    m = Y.shape[0]
    A0, A1, A2, A3 = cache["A0"], cache["A1"], cache["A2"], cache["A3"]
    Z1, Z2, Z3 = cache["Z1"], cache["Z2"], cache["Z3"]

    dZ4 = (y_hat - Y) / m
    dW4 = A3.T @ dZ4
    db4 = np.sum(dZ4, axis=0, keepdims=True)

    dA3 = dZ4 @ W4.T
    dZ3 = dA3 * relu_derivative(Z3)  
    dW3 = A2.T @ dZ3
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = A0.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def mae_metric(ds, y_hat, y, scale_back=False):
    if scale_back:
        return np.mean(np.abs((y_hat * ds.y_std + ds.y_mean) - (y * ds.y_std + ds.y_mean)))
    return np.mean(np.abs(y_hat - y))

def train(ds, epochs=10000, alpha=0.01) -> list:
    global W1, W2, W3, W4, b1, b2, b3, b4
    h_train_mse = []
    h_val_mse = []
    h_train_mae = []
    h_val_mae = []

    for e in range(epochs):
        y_hat, cache = feed_forward(ds.X_tren)
        loss_mse = cost(y_hat, ds.Y_tren_norm)
        
        y_val_hat, _ = feed_forward(ds.X_val)
        val_loss_mse = cost(y_val_hat, ds.Y_val_norm)

        train_mse = np.mean(np.abs(y_hat - ds.Y_tren_norm)) * ds.y_std
        val_mse = np.mean(np.abs(y_val_hat - ds.Y_val_norm)) * ds.y_std

        train_mae = np.mean(np.abs(y_hat - ds.Y_tren_norm)) * ds.y_std
        val_mae = np.mean(np.abs(y_val_hat - ds.Y_val_norm)) * ds.y_std

        h_train_mse.append(loss_mse)
        h_val_mse.append(val_loss_mse)
        h_train_mae.append(train_mae)
        h_val_mae.append(val_mae)

        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backprop(y_hat, ds.Y_tren_norm, cache)

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3
        W4 -= alpha * dW4
        b4 -= alpha * db4

        if e % 250 == 0:
            print(f"Epoch {e} | Train MSE: {train_mse:.0f}$ | Val MSE: {val_mse:.0f}$")
            print(f"Epoch {e} | Train MAE: {train_mae:.0f}$ | Val MAE: {val_mae:.0f}$")

    return h_train_mse, h_val_mse, h_train_mae, h_val_mae

def NN_from_scratch(ds, H1_size=64, H2_size=32, H3_size=16, epochs=10000, alpha=0.001):
    initialize_layers(H1_size, H2_size, H3_size)
    train_history_mse, val_history_mse, train_history_mae, val_history_mae = train(ds, epochs, alpha)
    y_test_pred = predict(ds, ds.X_test)

    mae_mse_chart(train_history_mse, val_history_mse, train_history_mae, val_history_mae)
    eval_chart(ds, y_test_pred)
    price_to_error_chart(ds, y_test_pred)


