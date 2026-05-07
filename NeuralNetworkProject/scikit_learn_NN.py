import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def MSE_chart(model) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(model.loss_curve_)
    plt.title("Krzywa uczenia - MLPRegressor (MSE)")
    plt.xlabel("Iteracje")
    plt.ylabel("Strata (Loss)")
    plt.grid(True)
    plt.show()

def MSE_MAE_error(ds, model) -> None:
    y_val_pred_norm = model.predict(ds.X_val)
    mse_val = mean_squared_error(ds.Y_val_norm, y_val_pred_norm)
    y_val_pred_dollars = y_val_pred_norm * ds.y_std + ds.y_mean
    mae_val = mean_absolute_error(ds.Y_val, y_val_pred_dollars)
    print(f"Błąd MSE (znormalizowany): {mse_val:.6f}")
    print(f"Błąd MAE (w dolarach): {mae_val:.2f} $")

def MSE_MAE_summary(ds, model) -> float:
    y_test_pred_norm = model.predict(ds.X_test)
    y_test_pred_dollars = y_test_pred_norm * ds.y_std + ds.y_mean
    mae_test = mean_absolute_error(ds.Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(ds.Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")
    return mae_test

def MLP_NN(ds, size_H1, size_H2, size_H3, _activation, _solver) -> float:
    MLP_regr = MLPRegressor(
        hidden_layer_sizes=(size_H1, size_H2, size_H3), # (128,64,16), (64,32,16), (64,32,8)
        activation=_activation, # logistic, relu
        solver=_solver, # adam, sgd
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=10000,          
        early_stopping=True,
        n_iter_no_change=50,    
        validation_fraction=0.1, 
        random_state=1,
        verbose=True            
    )

    MLP_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())
    
    MSE_chart(MLP_regr)
    MSE_MAE_error(ds, MLP_regr)
    mae_test = MSE_MAE_summary(ds, MLP_regr)
    return mae_test
    
# dodac mse
    

    

