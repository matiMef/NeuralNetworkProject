import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mse_mae_test(ds, model) -> tuple:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mse_test, mae_test

def tests_summary(all_tests) -> None:
    best_result_mse = float('inf')
    best_result_mae = float('inf')
    best_result_mse_idx = 999
    best_result_mae_idx = 999
    
    for i, result_dict in enumerate(all_tests):
        mse = result_dict.get("mse")
        mae = result_dict.get("mae")
    
    if mse < best_result_mse:
        best_result_mse = mse
        best_result_mse_idx = i
        
    if mae < best_result_mae:
        best_result_mae = mae
        best_result_mae_idx = i
    
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ DLA WSZYSTKICH TESTÓW (TEST SET) ---")
    print('Test:', best_result_mse_idx, 'Min MSE:', best_result_mse)
    print('Test:', best_result_mae_idx , 'Min MAE:', best_result_mae)

def training_MSE_chart(scikit_tests, loss_curves) -> None:
    plt.figure(figsize=(12, 7))
    for i, loss_curve in enumerate(loss_curves):
        label = f"Test {i}: {scikit_tests[i][0:3]} {scikit_tests[i][4]}"
        plt.plot(loss_curve, label=label)
    plt.title("Porównanie Krzywych Straty (MSE) dla wszystkich testów")
    plt.xlabel("Iteracje (Epoki)")
    plt.ylabel("Strata (Znormalizowana)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def training_R2_chart(scikit_tests, validation_scores) -> None:
    has_data = False
    plt.figure(figsize=(12, 7))

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

def loss_chart(h_train_mse, h_val_mse, h_train_mae, h_val_mae) -> None:
    fig, (ax_mse, ax_mae) = plt.subplots(1, 2, figsize=(16, 6))

    ax_mse.plot(h_train_mse, 'r-', label='Train MSE')
    ax_mse.plot(h_val_mse, 'b-', label='Val MSE')
    ax_mse.set_title("Koszt (MSE) - Normalizacja")
    ax_mse.set_xlabel("Epoki")
    ax_mse.grid(True, linestyle='--', alpha=0.7)
    ax_mse.legend()

    ax_mae.plot(h_train_mae, 'r--', label='Train MAE ($)')
    ax_mae.plot(h_val_mae, 'b--', label='Val MAE ($)')
    ax_mae.set_title("Błąd Średni (MAE) w Dolarach")
    ax_mae.set_xlabel("Epoki")
    ax_mae.grid(True, linestyle='--', alpha=0.7)
    ax_mae.legend()

    plt.tight_layout()
    plt.show()

def evaluation_chart(ds, y_test_pred) -> None:
    plt.title("Ewaluacja modelu na zbiorze testowym")
    plt.scatter(ds.Y_test, y_test_pred, alpha=0.5)
    plt.plot([ds.Y_test.min(), ds.Y_test.max()], [ds.Y_test.min(), ds.Y_test.max()], 'r--')
    plt.xlabel("Cena prawdziwa")
    plt.ylabel("Cena przewidziana")
    plt.show()

def category_price_chart(ds, y_test_pred) -> None:
    squared_errors = (y_test_pred - ds.Y_test)**2
    bins = np.arange(0, ds.Y_test.max() + 200000, 200000)
    bin_labels = [f"{int(b/1000)}k-{int((b+200000)/1000)}k" for b in bins[:-1]]
    df_error = pd.DataFrame({
        'Actual_Price': ds.Y_test.flatten(),
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

def summary_chart(all_tests, option="mse") -> None:
    categories = []
    values = []
    
    for i, result_dict in enumerate(all_tests):
        categories.append(result_dict.get("params") + f" {i}")
        values.append(result_dict.get(option))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    
    plt.figure(figsize=(12, 6))
    plt.bar(categories, values, color=colors)
    plt.title(f'Porównanie modeli za pomocą {option.upper()}')
    plt.xlabel('Parametry modelu')
    plt.xticks(rotation=30, fontsize="8")
    plt.ylabel(f'Metryka: {option.upper()}')
    plt.show()

def show_features_importance(model, option="Regresja Liniowa") -> None:
    features = ["sqft_living", "sqft_lot", "lat", "long", "floors", 
                "bedrooms", "bathrooms", "years_since_refurb", "house_age"]
    if option == "Regresja Liniowa":
        coefs = pd.Series(model.coef_, index=features).sort_values(ascending=False)
    else:
        coefs = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    colors = ['skyblue' if c > 0 else 'salmon' for c in coefs]
    
    plt.figure(figsize=(12, 6))
    plt.bar(coefs.index, coefs.values, color=colors)
    plt.axhline(0, color='black', linewidth=0.8) 
    plt.title(f"Istotność cech - {option} (Współczynniki)")
    plt.ylabel("Wartość współczynnika")
    plt.xlabel("Cechy")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def mse_chart(train_mse, val_mse) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(train_mse, 'r-', label='Train MSE')
    plt.plot(val_mse, 'b-', label='Val MSE')
    plt.title("Koszt (MSE) - Normalizacja")
    plt.legend()
    plt.show()