from sklearn.ensemble import RandomForestRegressor
from utils import MSE_MAE_val ,MSE_MAE_test

def random_forest(ds, _n_estimators, _max_depth) -> list:
    rf_regr = RandomForestRegressor(
        n_estimators=_n_estimators, 
        max_depth=_max_depth, 
        random_state=42,
        verbose=1 
    )

    rf_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())

    mse_val, mae_val = MSE_MAE_val(ds, rf_regr)
    mse_test, mae_test = MSE_MAE_test(ds, rf_regr)
    return mse_test, mae_test 