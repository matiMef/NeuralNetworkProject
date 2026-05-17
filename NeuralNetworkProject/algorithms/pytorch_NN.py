import numpy as np
import torch
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.utils import loss_chart, evaluation_chart, category_price_chart

MODEL_PATH  = "trained_pytorch_model.pt"
SCALER_PATH = "trained_pytorch_scaler.pkl"
FEATURES    = ["sqft_living", "sqft_lot", "lat", "long", "floors",
               "bedrooms", "bathrooms", "years_since_refurb", "house_age"]

W1 = W2 = W3 = W4 = b1 = b2 = b3 = b4 = None


def initialize_layers(H1_size: int, H2_size: int, H3_size: int) -> None:
    global W1, W2, W3, W4, b1, b2, b3, b4
    W1 = torch.randn(9,       H1_size) * np.sqrt(2. / 9)       ; W1.requires_grad_(True)
    b1 = torch.zeros(1,       H1_size)                          ; b1.requires_grad_(True)
    W2 = torch.randn(H1_size, H2_size) * np.sqrt(2. / H1_size) ; W2.requires_grad_(True)
    b2 = torch.zeros(1,       H2_size)                          ; b2.requires_grad_(True)
    W3 = torch.randn(H2_size, H3_size) * np.sqrt(2. / H2_size) ; W3.requires_grad_(True)
    b3 = torch.zeros(1,       H3_size)                          ; b3.requires_grad_(True)
    W4 = torch.randn(H3_size, 1)       * np.sqrt(2. / H3_size) ; W4.requires_grad_(True)
    b4 = torch.zeros(1,       1)                                ; b4.requires_grad_(True)


def relu(Z: torch.Tensor) -> torch.Tensor:
    return torch.clamp(Z, min=0)


def feed_forward(A0: torch.Tensor) -> torch.Tensor:
    Z1 = A0 @ W1 + b1 ; A1 = relu(Z1)
    Z2 = A1 @ W2 + b2 ; A2 = relu(Z2)
    Z3 = A2 @ W3 + b3 ; A3 = relu(Z3)
    Z4 = A3 @ W4 + b4
    return Z4


def cost(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    m = y.shape[0]
    return (1 / (2 * m)) * torch.sum((y_hat - y) ** 2)


def predict(ds, X: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        y_norm_pred = feed_forward(X)
    return y_norm_pred * ds.y_std + ds.y_mean


def train(ds, epochs: int = 10000, alpha: float = 0.01) -> tuple:
    params = [W1, b1, W2, b2, W3, b3, W4, b4]

    X_tr  = torch.tensor(ds.X_tren.astype(np.float32))
    Y_tr  = torch.tensor(ds.Y_tren_norm.astype(np.float32))
    X_val = torch.tensor(ds.X_val.astype(np.float32))
    Y_val = torch.tensor(ds.Y_val_norm.astype(np.float32))

    h_train_mse = []
    h_val_mse   = []
    h_train_mae = []
    h_val_mae   = []

    for e in range(epochs):
        y_hat    = feed_forward(X_tr)
        loss_mse = cost(y_hat, Y_tr)

        loss_mse.backward()

        with torch.no_grad():
            for p in params:
                p -= alpha * p.grad
                p.grad.zero_()

        with torch.no_grad():
            y_val_hat    = feed_forward(X_val)
            val_loss_mse = cost(y_val_hat, Y_val)
            train_mae    = torch.mean(torch.abs(y_hat     - Y_tr )).item() * ds.y_std
            val_mae      = torch.mean(torch.abs(y_val_hat - Y_val)).item() * ds.y_std

        h_train_mse.append(loss_mse.item())
        h_val_mse.append(val_loss_mse.item())
        h_train_mae.append(train_mae)
        h_val_mae.append(val_mae)

        if e % 250 == 0:
            print(f"Epoch {e} | Train MAE: {train_mae:.0f}$ | Val MAE: {val_mae:.0f}$")

    return h_train_mse, h_val_mse, h_train_mae, h_val_mae


def mse_mae_test(ds, model_name: str = "") -> tuple:
    X_test_t       = torch.tensor(ds.X_test.astype(np.float32))
    y_pred_dollars = predict(ds, X_test_t).numpy().flatten()
    mae_test       = mean_absolute_error(ds.Y_test.flatten(), y_pred_dollars)
    y_pred_norm    = (y_pred_dollars - ds.y_mean) / ds.y_std
    mse_test       = mean_squared_error(ds.Y_test_norm.flatten(), y_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    if model_name:
        print(f"Model: {model_name}")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mse_test, mae_test


def NN_from_scratch(ds, H1_size: int = 64, H2_size: int = 32,
                    H3_size: int = 16, epochs: int = 10000,
                    alpha: float = 0.001, save_model: bool = False) -> tuple:
    model_name = f"Sieć neuronowa from scratch (PyTorch) | Warstwy: {H1_size}-{H2_size}-{H3_size} | Epoki: {epochs} | LR: {alpha}"

    initialize_layers(H1_size, H2_size, H3_size)
    train_mse, val_mse, train_mae, val_mae = train(ds, epochs, alpha)

    X_test_t    = torch.tensor(ds.X_test.astype(np.float32))
    y_test_pred = predict(ds, X_test_t).numpy().reshape(-1, 1)

    mse_test, mae_test = mse_mae_test(ds, model_name)
    loss_chart(train_mse, val_mse, train_mae, val_mae, model_name)
    evaluation_chart(ds, y_test_pred, model_name)
    category_price_chart(ds, y_test_pred, model_name)

    if save_model:
        _save_model(ds, H1_size, H2_size, H3_size)

    return mse_test, mae_test


def _save_model(ds, H1_size: int, H2_size: int, H3_size: int) -> None:
    weights = {
        'W1': W1.detach(), 'b1': b1.detach(),
        'W2': W2.detach(), 'b2': b2.detach(),
        'W3': W3.detach(), 'b3': b3.detach(),
        'W4': W4.detach(), 'b4': b4.detach(),
    }
    torch.save(weights, MODEL_PATH)
    scaler = {
        'X_mean':  ds.X_mean.values,
        'X_std':   ds.X_std.values,
        'y_mean':  float(ds.y_mean),
        'y_std':   float(ds.y_std),
        'H1_size': H1_size,
        'H2_size': H2_size,
        'H3_size': H3_size,
    }
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Zapis] Model  → {MODEL_PATH}")
    print(f"[Zapis] Scaler → {SCALER_PATH}")


def load_model() -> dict:
    global W1, W2, W3, W4, b1, b2, b3, b4
    weights = torch.load(MODEL_PATH, map_location='cpu')
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']
    W3, b3 = weights['W3'], weights['b3']
    W4, b4 = weights['W4'], weights['b4']
    return joblib.load(SCALER_PATH)


def predict_price(feature_values: dict) -> float:
    scaler = load_model()
    x      = np.array([feature_values[f] for f in FEATURES], dtype=np.float32)
    x_norm = (x - scaler['X_mean']) / scaler['X_std']
    x_t    = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        y_norm = feed_forward(x_t).item()
    return y_norm * scaler['y_std'] + scaler['y_mean']