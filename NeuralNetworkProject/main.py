from utils.dataset import Dataset
from utils.utils import training_MSE_chart, training_R2_chart, tests_summary, summary_chart
from algorithms.from_scratch import NN_from_scratch
from algorithms.pytorch_NN import NN_from_scratch as pytorch_NN_from_scratch
from algorithms.scikit_learn_NN import MLP_NN
from algorithms.scikit_learn_LR import linear_regression
from algorithms.scikit_learn_RF import random_forest

def main() -> None:
    ds = Dataset('utils/kc_house_data.csv')
    loss_curves = []
    validation_scores = []
    all_tests = []

    from_scratch_tests = [
        [128, 64, 16, 1000, 0.001],
        [128, 32, 16, 1000, 0.001],
        [64, 32, 16, 1000, 0.001],
        [64, 32, 8, 1000, 0.001]
    ]

    pytorch_tests = [
        [128, 64, 16, 1000, 0.001, False],
        [128, 32, 16, 1000, 0.001, False],
        [64,  32, 16, 1000, 0.001, False],
        [64,  32,  8, 1000, 0.001, True],
    ]

    scikit_NN_tests = [
        [128, 64, 32, 'logistic', 'adam', 10000, 0.0001],
        [64, 32, 16, 'logistic', 'adam', 10000, 0.0001],
        [64, 32, 8, 'logistic', 'adam', 10000, 0.0001],
        [128, 64, 32, 'relu', 'adam', 10000, 0.0001],
        [64, 32, 16, 'relu', 'adam', 10000, 0.0001],
        [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        [64, 32, 8,  'relu', 'adam', 10000, 0.0001],
        [64, 32, 16, 'logistic', 'sgd', 10000, 0.0001],
        [64, 32, 8, 'logistic', 'sgd', 10000, 0.0001],
        [128, 64, 32, 'relu', 'sgd', 10000, 0.0001],
        [64, 32, 16, 'relu', 'sgd', 10000, 0.0001],
        [64, 32, 8,  'relu', 'sgd', 10000, 0.0001]
    ]

    scikit_RF_tests = [
        [500, 30],
        [425, 25],
        [250, 20],
        [200, 20],
        [400, 25],
        [300, 30]
    ]

    for test in from_scratch_tests:
        mse_result, mae_result = NN_from_scratch(ds, test[0], test[1], test[2], test[3], test[4])
        all_tests.append({
            "params": f"Scratch | Layers: {test[0]}, {test[1]}, {test[2]}",
            "mse": mse_result,
            "mae": mae_result
        })

    for test in pytorch_tests:
        mse_result, mae_result = pytorch_NN_from_scratch(
            ds, test[0], test[1], test[2], test[3], test[4], save_model=test[5]
        )
        all_tests.append({
            "params": f"PyTorch | Layers: {test[0]}, {test[1]}, {test[2]}",
            "mse": mse_result,
            "mae": mae_result
        })

    for test in scikit_NN_tests:
        mse_result, mae_result, loss_curve, validation_score = MLP_NN(ds, test[0], test[1], test[2], test[3], test[4], test[5], test[6])
        loss_curves.append(loss_curve)
        validation_scores.append(validation_score)
        all_tests.append({
            "params": f"Activ: {test[3]}, Solver: {test[4]}",
            "mse": mse_result,
            "mae": mae_result
        })

    print(f"\n--- WYNIKI EWALUACJI KOŃCOWEJ DLA TESTÓW SIECI NEURONOWYCH (TEST SET) ---")
    training_MSE_chart(scikit_NN_tests, loss_curves)
    training_R2_chart(scikit_NN_tests, validation_scores)

    mse_result, mae_result = linear_regression(ds)
    all_tests.append({
        "params": f"Linear Regression",
        "mse": mse_result,
        "mae": mae_result
    })

    for test in scikit_RF_tests:
        mse_result, mae_result = random_forest(ds, test[0], test[1])
        all_tests.append({
            "params": f"N est: {test[0]}, Max depth: {test[1]}",
            "mse": mse_result,
            "mae": mae_result,
        })

    tests_summary(all_tests)
    summary_chart(all_tests, "mse")
    summary_chart(all_tests, "mae")

if __name__ == "__main__":
    main()