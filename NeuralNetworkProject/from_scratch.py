import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Przygotowanie danych ---
current_year = 2026
# Zakładamy, że plik kc_house_data.csv jest w tym samym folderze
data = pd.read_csv('kc_house_data.csv')
data = data[["sqft_living", "sqft_lot", "lat", "long", "yr_built", "yr_renovated", "floors", "bedrooms", "bathrooms", "price"]]

# Filtrowanie i clipping
data = data[data["bedrooms"] <= 15]
data["sqft_lot"] = data["sqft_lot"].clip(upper=213008)
data["sqft_living"] = data["sqft_living"].clip(upper=4978)

# Nowe cechy
last_refurb_year = data["yr_renovated"].mask(data["yr_renovated"] == 0, data["yr_built"])
data["years_since_refurb"] = current_year - last_refurb_year
data["house_age"] = current_year - data["yr_built"]

data = data.drop(["yr_renovated", "yr_built"], axis=1)

# Normalizacja X
X = data.drop("price", axis=1)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

X_np = X_normalized.values
Y_np = data["price"].values.reshape(-1, 1) # Od razu do (m, 1)

# Podział na zbiory
np.random.seed(42)
indices = np.random.permutation(len(X_np))
tren_size = int(0.8 * len(X_np))

X_tren = X_np[indices[:tren_size]]
Y_tren = Y_np[indices[:tren_size]]

# Normalizacja Y (Kluczowe dla stabilności sieci)
y_mean = Y_tren.mean()
y_std = Y_tren.std()
Y_tren_norm = (Y_tren - y_mean) / y_std

# --- 2. Inicjalizacja Sieci ---
# Używamy lepszej inicjalizacji (He initialization)
W1 = np.random.randn(9, 64) * np.sqrt(2./9)
b1 = np.zeros((1, 64))
W2 = np.random.randn(64, 32) * np.sqrt(2./64)
b2 = np.zeros((1, 32))
W3 = np.random.randn(32, 1) * np.sqrt(2./32)
b3 = np.zeros((1, 1))

# --- 3. Funkcje Pomocnicze ---
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
    cache = {"A0": A0, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return Z3, cache

def backprop(y_hat, Y, cache):
    global W1, W2, W3, b1, b2, b3
    m = Y.shape[0]
    A0, A1, A2 = cache["A0"], cache["A1"], cache["A2"]
    Z1, Z2 = cache["Z1"], cache["Z2"]

    dZ3 = (y_hat - Y) / m
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

    return dW1, db1, dW2, db2, dW3, db3

# --- 4. Główna Pętla Treningowa ---
def train(epochs=50000, alpha=0.01):
    global W1, W2, W3, b1, b2, b3
    history = []

    plt.ion() # Włączenie trybu interaktywnego dla wykresu
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'r-')
    ax.set_title("Koszt w czasie rzeczywistym")
    ax.set_xlabel("Epoka")
    ax.set_ylabel("MSE")

    for e in range(epochs):
        y_hat, cache = feed_forward(X_tren)
        loss = cost(y_hat, Y_tren_norm)
        history.append(loss)

        dW1, db1, dW2, db2, dW3, db3 = backprop(y_hat, Y_tren_norm, cache)

        # Aktualizacja wag
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
        W3 -= alpha * dW3
        b3 -= alpha * db3

        if e % 10 == 0:
            # Aktualizacja wykresu
            line.set_xdata(range(len(history)))
            line.set_ydata(history)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)
            if e % 100 == 0:
                print(f"Epoch {e}, Cost: {loss:.6f}")

    plt.ioff() # Wyłączenie trybu interaktywnego
    plt.show()
    return history

# --- 5. Wywołanie ---
costs = train(epochs=50000, alpha=0.01)

