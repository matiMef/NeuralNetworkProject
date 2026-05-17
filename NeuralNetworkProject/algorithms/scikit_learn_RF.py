from sklearn.ensemble import RandomForestRegressor
from utils.utils import mse_mae_test, evaluation_chart, show_features_importance

def random_forest(ds, _n_estimators, _max_depth) -> tuple:
    model_name = f"Random Forest (Scikit-learn) | Drzewa: {_n_estimators} | Max głębokość: {_max_depth}"

    rf_regr = RandomForestRegressor(
        n_estimators=_n_estimators, 
        max_depth=_max_depth, 
        random_state=42,
        verbose=1 
    )

    rf_regr.fit(ds.X_tren, ds.Y_tren_norm.ravel())

    mse_test, mae_test = mse_mae_test(ds, rf_regr, model_name)
    y_pred_mlp_norm = rf_regr.predict(ds.X_test)
    y_pred_mlp_dollars = y_pred_mlp_norm * ds.y_std + ds.y_mean

    show_features_importance(rf_regr, "Random Forest")
    evaluation_chart(ds, y_pred_mlp_dollars, model_name)

    return mse_test, mae_test