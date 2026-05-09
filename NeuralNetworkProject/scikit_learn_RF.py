from sklearn.ensemble import RandomForestRegressor
from utils import MSE_MAE_summary

def random_forest(ds) -> None:
    rf_regr = RandomForestRegressor(
        n_estimators=500, 
        max_depth=30, 
        random_state=42,
        verbose=1 
    )

    rf_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())

    MSE_MAE_summary(ds, rf_regr)