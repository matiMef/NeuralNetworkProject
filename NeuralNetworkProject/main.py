from dataset import Dataset
from from_scratch import NN_from_scratch
from scikit_learn_NN import MLP_NN
from scikit_learn_LR import linear_regression
from scikit_learn_RF import random_forest
from utils import MSE_comparison, MAE_comparison, MLP_NN_traning_MSE_chart, MLP_NN_traning_R2_chart

def all_tests_summary(all_tests) -> None:
    best_result_mse = float('inf')
    best_result_mae = float('inf')
    best_result_mse_idx = 999
    best_result_mae_idx = 999
    
    for result in range (len(all_tests)):
        result_dict = all_tests[result]
        mse = result_dict.get("mse")
        if mse < best_result_mse:
            best_result_mse = mse
            best_result_mse_idx = result
    
    for result in range (len(all_tests)):
        result_dict = all_tests[result]
        mae= result_dict.get("mae")
        if mse < best_result_mae :
            best_result_mae  = mae
            best_result_mae_idx  = result
    
    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ DLA WSZYSTKICH TESTÓW (TEST SET) ---")
    print('Test: ', best_result_mse_idx, 'Min MSE: ', best_result_mse)
    print('Test: ', best_result_mae_idx , 'Min MAE: ', best_result_mae)

def main() -> None:
    ds = Dataset('kc_house_data.csv')

    # NN_from_scratch(ds, 1000, 0.001)

    scikit_NN_mse_results = []
    scikit_NN_mae_results = []
    loss_curves = []
    validation_scores = []
    scikit_LR_mse_results = []
    scikit_LR_mae_results = []
    scikit_RF_mse_results = []
    scikit_RF_mae_results = []
    all_tests = []
    test_dict = dict(params = "", mse = 0, mae = 0)
    
    # H1 size, H2 size, H3 size, model, solver, max_iter, alpha
    scikit_NN_tests = [
        [128, 64, 32, 'logistic', 'adam', 10000, 0.0001],
        [64, 32, 16, 'logistic', 'adam', 10000, 0.0001],
        [64, 32, 8, 'logistic', 'adam', 10000, 0.0001],
        [128, 64, 32, 'relu', 'adam', 10000, 0.0001],
        [64, 32, 16, 'relu', 'adam', 10000, 0.0001],
        [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        # [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        # [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        # [64, 32, 16, 'logistic', 'sgd', 10000, 0.0001],
        # [64, 32, 8, 'logistic', 'sgd', 10000, 0.0001],
        # [128, 64, 32, 'relu', 'sgd', 10000, 0.0001],
        # [64, 32, 16, 'relu', 'sgd', 10000, 0.0001],
        # [64, 32, 8,  'relu', 'sgd', 10000, 0.0001]
    ]

    scikit_RF_tests = [
        [500, 30],
        [425, 25],
        # [250, 20],
        # [200, 20],
        # [400, 25],
        [300, 30]
    ]

    for test in scikit_NN_tests:
        mse_result, mae_result, loss_curve, validation_score = MLP_NN(ds, test[0], test[1], test[2], test[3], test[4], test[5], test[6])
        scikit_NN_mse_results.append(mse_result)
        scikit_NN_mae_results.append(mae_result)
        loss_curves.append(loss_curve)
        validation_scores.append(validation_score)

        current_result = {
        "params": f"Layers: {test[0:3]}, Activ: {test[3]}, Solver: {test[4]}, Alpha: {test[6]}",
        "mse": mse_result,
        "mae": mae_result
        }
        all_tests.append(current_result)
    
    MSE_comparison(scikit_NN_mse_results)
    MAE_comparison(scikit_NN_mae_results)
    MLP_NN_traning_MSE_chart(scikit_NN_tests, loss_curves)
    MLP_NN_traning_R2_chart(scikit_NN_tests, validation_scores)
    
    mse_result, mae_result = linear_regression(ds)
    scikit_LR_mse_results.append(mse_result)
    scikit_LR_mae_results.append(mae_result)
    current_result = {
        "params": f"Linear Regression",
        "mse": mse_result,
        "mae": mae_result
    }
    all_tests.append(current_result)
    
    for test in scikit_RF_tests:
        mse_result, mae_result = random_forest(ds, test[0], test[1])
        scikit_RF_mse_results.append(mse_result)
        scikit_RF_mae_results.append(mae_result)
        current_result = {
        "params": f"N est: {test[0]}, Max depth: {test[1]}",
        "mse": mse_result,
        "mae": mae_result
    }
    all_tests.append(current_result)

    MSE_comparison(scikit_RF_mse_results)
    MAE_comparison(scikit_RF_mae_results)

    all_tests_summary(all_tests)
    
if __name__ == "__main__":
    main()

# Dodać do modelu dane walidacyjne
# Dodać funckje dla walidacji
# Dodać wykresy słupkowe dla testu i walidacji dla mse i dla mae
# Dodać wykresy dla ewaluacji
# Poprawić funckje sieci from scratch, żeby przyjmowała argumenty
# Dodać benchmarki czasowe
# Dodać wykresy czasów
# Dodać model Pytroch