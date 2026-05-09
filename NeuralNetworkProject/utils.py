from sklearn.metrics import mean_squared_error, mean_absolute_error

def MSE_MAE_summary(ds, model) -> float:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mae_test
