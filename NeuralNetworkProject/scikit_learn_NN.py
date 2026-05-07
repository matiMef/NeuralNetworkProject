def neural_network(size_H1, size_H2, size_H3, _activation, _solver):
    
    regr = MLPRegressor(
        hidden_layer_sizes=(size_H1, size_H2, size_H3), # (128,64,32), (128,64,16), (64,32,16), (64,32,8)
        activation=_activation, # logistic, relu, tanh
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

    regr.fit(X_tren, Y_tren_norm.ravel())
    plt.figure(figsize=(8, 5))
    plt.plot(regr.loss_curve_)
    plt.title("Krzywa uczenia - MLPRegressor (MSE)")
    plt.xlabel("Iteracje")
    plt.ylabel("Strata (Loss)")
    plt.grid(True)
    plt.show()

    y_val_pred_norm = regr.predict(X_val)
    mse_val = mean_squared_error(Y_val_norm, y_val_pred_norm)
    y_val_pred_dollars = y_val_pred_norm * y_std + y_mean
    mae_val = mean_absolute_error(Y_val, y_val_pred_dollars)
    print(f"Błąd MSE (znormalizowany): {mse_val:.6f}")
    print(f"Błąd MAE (w dolarach): {mae_val:.2f} $")

    y_test_pred_norm = regr.predict(X_test)
    y_test_pred_dollars = y_test_pred_norm * y_std + y_mean
    mae_test = mean_absolute_error(Y_test, y_test_pred_dollars)
    mse_test = mean_squared_error(Y_test_norm, y_test_pred_norm)
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ (TEST SET) ---")
    print(f"Błąd MSE (znormalizowany): {mse_test:.6f}")
    print(f"Błąd MAE: {mae_test:.2f} $")