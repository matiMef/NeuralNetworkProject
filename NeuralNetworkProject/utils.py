import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def MSE_MAE_error(ds, model) -> None:
    y_val_pred_norm = model.predict(ds.X_val)
    mse_val = mean_squared_error(ds.Y_val_norm, y_val_pred_norm)
    y_val_pred_dollars = y_val_pred_norm * ds.y_std + ds.y_mean
    mae_val = mean_absolute_error(ds.Y_val, y_val_pred_dollars)
    print(f"Błąd MSE (znormalizowany): {mse_val:.6f}")
    print(f"Błąd MAE (w dolarach): {mae_val:.2f} $")

def MSE_MAE_summary(ds, model) -> list:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mse_test, mae_test

def MSE_comparison(scikit_results) -> None:
    best_result = float('inf')
    best_result_idx = 999
    for result in range (len(scikit_results)):
        if scikit_results[result] < best_result:
            best_result = scikit_results[result]
            best_result_idx = result
    print('Test: ', best_result_idx, 'Min MAE: ', best_result)

def MAE_comparison(scikit_results) -> None:
    best_result = float('inf')
    best_result_idx = 999
    for result in range (len(scikit_results)):
        if scikit_results[result] < best_result:
            best_result = scikit_results[result]
            best_result_idx = result
    print('Test: ', best_result_idx, 'Min MSE: ', best_result)

def MLP_NN_traning_MSE_chart(scikit_tests, loss_curves) -> None:
    plt.figure(figsize=(12, 7))
    for i, loss_curve in enumerate(loss_curves):
        label = f"Test {i}: {scikit_tests[i][0:3]} {scikit_tests[i][4]}"
        plt.plot(loss_curve, label=label)
    
    plt.title("Porównanie Krzywych Straty (MSE) dla wszystkich testów")
    plt.xlabel("Iteracje (Epoki)")
    plt.ylabel("Strata (Znormalizowana)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Legenda poza wykresem
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def MLP_NN_traning_R2_chart(scikit_tests, validation_scores) -> None:
    plt.figure(figsize=(12, 7))
    has_data = False
    for i, v_score in enumerate(validation_scores):
        if len(v_score) > 0:
            label = f"Test {i}: {scikit_tests[i][3]}"
            plt.plot(v_score, label=label)
            has_data = True
    
    if has_data:
        plt.title("Porównanie Wyników Walidacji ($R^2$) dla wszystkich testów")
        plt.xlabel("Iteracje (Epoki)")
        plt.ylabel("Współczynnik $R^2$")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "Brak danych walidacji (early_stopping=False)", 
                 ha='center', va='center')
        
    plt.tight_layout()
    plt.show()