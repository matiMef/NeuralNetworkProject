import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

W1 = np.random.randn(9, 128) * np.sqrt(2./9)
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 32) * np.sqrt(2./128)
b2 = np.zeros((1, 32))
W3 = np.random.randn(32, 16) * np.sqrt(2./32) # Nowa warstwa
b3 = np.zeros((1, 16))
W4 = np.random.randn(16, 1) * np.sqrt(2./16)  # Warstwa wyjściowa
b4 = np.zeros((1, 1))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def cost(y_hat, y):
    m = y.shape[0]
    return (1 / (2 * m)) * np.sum((y_hat - y)**2)

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

def mae_metric(y_hat, y, scale_back=False):
    if scale_back:
        return np.mean(np.abs((y_hat * y_std + y_mean) - (y * y_std + y_mean)))
    return np.mean(np.abs(y_hat - y))

def train(epochs=10000, alpha=0.01):
    global W1, W2, W3, W4, b1, b2, b3, b4
    h_train_mse = []
    h_val_mse = []
    h_train_mae = []
    h_val_mae = []

    plt.ion()
    fig, (ax_mse, ax_mae) = plt.subplots(1, 2, figsize=(16, 6))

    l_train_mse, = ax_mse.plot([], [], 'r-', label='Train MSE')
    l_val_mse, = ax_mse.plot([], [], 'b-', label='Val MSE')
    ax_mse.set_title("Koszt (MSE) - Normalizacja")
    ax_mse.legend()

    l_train_mae, = ax_mae.plot([], [], 'r--', label='Train MAE ($)')
    l_val_mae, = ax_mae.plot([], [], 'b--', label='Val MAE ($)')
    ax_mae.set_title("Błąd Średni (MAE) w Dolarach")
    ax_mae.legend()

    for e in range(epochs):
        y_hat, cache = feed_forward(X_tren)
        loss_mse = cost(y_hat, Y_tren_norm)
        
        y_val_hat, _ = feed_forward(X_val)
        val_loss_mse = cost(y_val_hat, Y_val_norm)

        train_mae = np.mean(np.abs(y_hat - Y_tren_norm)) * y_std
        val_mae = np.mean(np.abs(y_val_hat - Y_val_norm)) * y_std

        # Zapisywanie historii
        h_train_mse.append(loss_mse)
        h_val_mse.append(val_loss_mse)
        h_train_mae.append(train_mae)
        h_val_mae.append(val_mae)

        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backprop(y_hat, Y_tren_norm, cache)

        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3
        W4 -= alpha * dW4
        b4 -= alpha * db4

        if e % 100 == 0:
            l_train_mse.set_data(range(len(h_train_mse)), h_train_mse)
            l_val_mse.set_data(range(len(h_val_mse)), h_val_mse)
            ax_mse.relim()
            ax_mse.autoscale_view()

            l_train_mae.set_data(range(len(h_train_mae)), h_train_mae)
            l_val_mae.set_data(range(len(h_val_mae)), h_val_mae)
            ax_mae.relim()
            ax_mae.autoscale_view()

            plt.pause(0.01)
            print(f"Epoch {e} | Train MAE: {train_mae:.0f}$ | Val MAE: {val_mae:.0f}$")

    plt.ioff()
    plt.show()
    return h_train_mse, h_val_mse

def predict(X):
    y_norm_pred, _ = feed_forward(X)
    return y_norm_pred * y_std + y_mean

train_history, val_history = train(epochs=25000, alpha=0.001)
y_test_pred = predict(X_test)
mae = np.mean(np.abs(y_test_pred - Y_test))
print(f"\n--- WYNIK KOŃCOWY ---")
print(f"Średni błąd (MAE): {mae:.2f} $")

plt.scatter(Y_test, y_test_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel("Cena prawdziwa")
plt.ylabel("Cena przewidziana")
plt.show()

squared_errors = (y_test_pred - Y_test)**2

bins = np.arange(0, Y_test.max() + 200000, 200000)
bin_labels = [f"{int(b/1000)}k-{int((b+200000)/1000)}k" for b in bins[:-1]]

df_error = pd.DataFrame({
    'Actual_Price': Y_test.flatten(),
    'Squared_Error': squared_errors.flatten()
})

df_error['Price_Bin'] = pd.cut(df_error['Actual_Price'], bins=bins, labels=bin_labels)


mse_per_bin = df_error.groupby('Price_Bin', observed=False)['Squared_Error'].mean()

plt.figure(figsize=(12, 6))
mse_per_bin.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Średni błąd MSE w zależności od przedziału cenowego nieruchomości")
plt.xlabel("Przedział cenowy domu [$]")
plt.ylabel("Średni błąd kwadratowy (MSE)")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()