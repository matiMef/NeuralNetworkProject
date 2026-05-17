import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mse_mae_test(ds, model, model_name: str = "") -> tuple:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    if model_name:
        print(f"Model: {model_name}")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mse_test, mae_test

def tests_summary(all_tests) -> None:
    best_result_mse = float('inf')
    best_result_mae = float('inf')
    best_result_mse_idx = 0
    best_result_mae_idx = 0

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
    print(f"Najlepszy MSE → Test {best_result_mse_idx}: {all_tests[best_result_mse_idx]['params']} | MSE: {best_result_mse:.6f}")
    print(f"Najlepszy MAE → Test {best_result_mae_idx}: {all_tests[best_result_mae_idx]['params']} | MAE: {best_result_mae:.2f} $")

def training_MSE_chart(scikit_tests, loss_curves) -> None:
    plt.figure(figsize=(12, 7))
    for i, loss_curve in enumerate(loss_curves):
        label = f"Test {i}: {scikit_tests[i][0:3]} {scikit_tests[i][4]}"
        plt.plot(loss_curve, label=label)
    plt.title("Scikit-learn MLP – Porównanie krzywych straty (MSE) dla wszystkich testów")
    plt.xlabel("Iteracje (Epoki)")
    plt.ylabel("Strata MSE (znormalizowana)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        plt.title("Scikit-learn MLP – Porównanie wyników walidacji ($R^2$) dla wszystkich testów")
        plt.xlabel("Iteracje (Epoki)")
        plt.ylabel("Współczynnik $R^2$")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, "Brak danych walidacji (early_stopping=False)", 
                 ha='center', va='center')
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def loss_chart(h_train_mse, h_val_mse, h_train_mae, h_val_mae, model_name: str = "") -> None:
    fig, (ax_mse, ax_mae) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(model_name if model_name else "", fontsize=11, fontweight='bold')

    ax_mse.plot(h_train_mse, 'r-', label='Train MSE')
    ax_mse.plot(h_val_mse,   'b-', label='Val MSE')
    ax_mse.set_title("Krzywa uczenia – strata MSE (znormalizowana)", fontsize=10, color='gray')
    ax_mse.set_xlabel("Epoki")
    ax_mse.set_ylabel("Strata MSE (znormalizowana)")
    ax_mse.grid(True, linestyle='--', alpha=0.7)
    ax_mse.legend(loc='upper right')

    ax_mae.plot(h_train_mae, 'r--', label='Train MAE')
    ax_mae.plot(h_val_mae,   'b--', label='Val MAE')
    ax_mae.set_title("Krzywa uczenia – błąd MAE w dolarach", fontsize=10, color='gray')
    ax_mae.set_xlabel("Epoki")
    ax_mae.set_ylabel("Błąd MAE [$]")
    ax_mae.grid(True, linestyle='--', alpha=0.7)
    ax_mae.legend(loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def evaluation_chart(ds, y_test_pred, model_name: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(model_name if model_name else "",
                 fontsize=11, fontweight='bold')
    ax.set_title("Cena przewidziana vs rzeczywista – zbiór testowy", fontsize=10, color='gray')
    ax.scatter(ds.Y_test, y_test_pred, alpha=0.5, label='Predykcje')
    ax.plot([ds.Y_test.min(), ds.Y_test.max()],
            [ds.Y_test.min(), ds.Y_test.max()], 'r--', label='Idealna predykcja (y=x)')
    ax.set_xlabel("Cena rzeczywista [$]")
    ax.set_ylabel("Cena przewidziana [$]")
    ax.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def category_price_chart(ds, y_test_pred, model_name: str = "") -> None:
    squared_errors = (y_test_pred - ds.Y_test)**2
    bins = np.arange(0, ds.Y_test.max() + 200000, 200000)
    bin_labels = [f"{int(b/1000)}k-{int((b+200000)/1000)}k" for b in bins[:-1]]
    df_error = pd.DataFrame({
        'Actual_Price': ds.Y_test.flatten(),
        'Squared_Error': squared_errors.flatten()
    })
    df_error['Price_Bin'] = pd.cut(df_error['Actual_Price'], bins=bins, labels=bin_labels)
    mse_per_bin = df_error.groupby('Price_Bin', observed=False)['Squared_Error'].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(model_name if model_name else "",
                 fontsize=11, fontweight='bold')
    ax.set_title("Błąd MSE w zależności od przedziału cenowego nieruchomości – zbiór testowy",
                 fontsize=10, color='gray')
    mse_per_bin.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_xlabel("Przedział cenowy nieruchomości [$]")
    ax.set_ylabel("Średni błąd kwadratowy MSE")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def summary_chart(all_tests, option="mse") -> None:
    import matplotlib.patches as mpatches

    labels = [result_dict.get("params") for result_dict in all_tests]
    values = [result_dict.get(option)   for result_dict in all_tests]
    n      = len(labels)

    colors  = plt.cm.tab20(np.linspace(0, 1, n))
    x_pos   = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(x_pos, values, color=colors, edgecolor='white', linewidth=0.6)

    ax.set_title(f'Porównanie wszystkich modeli – metryka {option.upper()}', fontsize=13)
    ax.set_xlabel('Numer testu', fontsize=11)
    ax.set_ylabel(f'Wartość metryki {option.upper()}', fontsize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in range(n)], fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    patches = [mpatches.Patch(color=colors[i], label=f"{i}: {labels[i]}") for i in range(n)]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7.5, framealpha=0.9, title="Modele")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
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
    plt.title(f"Istotność cech – {option}")
    plt.ylabel("Wartość współczynnika")
    plt.xlabel("Cechy wejściowe")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def mse_chart(train_mse, val_mse, model_name: str = "") -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle(model_name if model_name else "",
                 fontsize=11, fontweight='bold')
    ax.set_title("Krzywa uczenia – strata MSE (zbiór treningowy vs walidacyjny)",
                 fontsize=10, color='gray')
    ax.plot(train_mse, 'r-',  label='Train MSE (zbiór treningowy)')
    ax.plot(val_mse,   'b-',  label='Val MSE (zbiór walidacyjny)')
    ax.set_xlabel("Epoki")
    ax.set_ylabel("Strata MSE (znormalizowana)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()