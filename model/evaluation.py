import pickle
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    data_dir = root_path / "data"
    pickle_dir = root_path / "pickled"
    path_results_folder = Path(__file__).parent / "results"

    preprocessed_all_features_path = data_dir / "preprocessed_cancer_data_all_features.csv"
    preprocessed_pca_path = data_dir / "preprocessed_cancer_data_pca.csv"

    for test_on in ["all"]:  # "pca"

        # Load the data
        try:
            if test_on == "pca":
                data = pd.read_csv(preprocessed_pca_path)
            elif test_on == "all":
                data = pd.read_csv(preprocessed_all_features_path)
        except NameError:
            if test_on == "pca":
                data = pd.read_csv(root_path / "preprocessed_cancer_data_pca.csv")
            elif test_on == "all":
                data = pd.read_csv(root_path / "preprocessed_cancer_data_all_features.csv")

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        # Tune hyperparameters using GridSearchCV
        params = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
        grid_log_reg = GridSearchCV(log_reg, param_grid=params, scoring='f1', cv=5)
        grid_log_reg.fit(X_train, y_train)
        # Best model
        log_reg_best = grid_log_reg.best_estimator_

        # save the trained model to a pickle file
        if test_on == "pca":
            with open(pickle_dir / "log_reg_model_pca.pkl", 'wb') as f:
                pickle.dump(log_reg_best, f)
        elif test_on == "all":
            with open(pickle_dir / "log_reg_model_all_features.pkl", 'wb') as f:
                pickle.dump(log_reg_best, f)

        # svc = SVC(random_state=42)
        # # Tune hyperparameters using GridSearchCV
        # params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        # grid_svc = GridSearchCV(svc, param_grid=params, scoring='f1', cv=5)
        # grid_svc.fit(X_train, y_train)
        # # Best model
        # svc_best = grid_svc.best_estimator_

        # # save the trained model to a pickle file
        # if test_on == "pca":
        #     with open(pickle_dir / "svc_model_pca.pkl", 'wb') as f:
        #         pickle.dump(svc_best, f)
        # elif test_on == "all":
        #     with open(pickle_dir / "svc_model_all_features.pkl", 'wb') as f:
        #         pickle.dump(svc_best, f)

        # rfc = RandomForestClassifier(random_state=42)
        # # Tune hyperparameters using GridSearchCV
        # params = {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]}
        # grid_rfc = GridSearchCV(rfc, param_grid=params, scoring='f1', cv=5)
        # grid_rfc.fit(X_train, y_train)
        # # Best model
        # rfc_best = grid_rfc.best_estimator_

        # # save the trained model to a pickle file
        # if test_on == "pca":
        #     with open(pickle_dir / "rfc_model_pca.pkl", 'wb') as f:
        #         pickle.dump(rfc_best, f)
        # elif test_on == "all":
        #     with open(pickle_dir / "rfc_model_all_features.pkl", 'wb') as f:
        #         pickle.dump(rfc_best, f)

        # gbc = GradientBoostingClassifier(random_state=42)
        # # Tune hyperparameters using GridSearchCV
        # params = {"n_estimators": [100, 200, 300], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
        # grid_gbc = GridSearchCV(gbc, param_grid=params, scoring='f1', cv=5)
        # grid_gbc.fit(X_train, y_train)
        # # Best model
        # gbc_best = grid_gbc.best_estimator_

        # # save the trained model to a pickle file
        # if test_on == "pca":
        #     with open(pickle_dir / "gbc_model_pca.pkl", 'wb') as f:
        #         pickle.dump(gbc_best, f)
        # elif test_on == "all":
        #     with open(pickle_dir / "gbc_model_all_features.pkl", 'wb') as f:
        #         pickle.dump(gbc_best, f)

        # # Evaluate models
        log_reg_eval = evaluate_model(log_reg_best, X_test, y_test)
        # svc_eval = evaluate_model(svc_best, X_test, y_test)
        # rfc_eval = evaluate_model(rfc_best, X_test, y_test)
        # gbc_eval = evaluate_model(gbc_best, X_test, y_test)

        # Print results
        print("Logistic Regression Performance:", log_reg_eval)
        # print("Support Vector Machine Performance:", svc_eval)
        # print("Random Forest Performance:", rfc_eval)
        # print("Gradient Boosting Performance:", gbc_eval)

        # save performances:
        with open(path_results_folder / f"{test_on}_log_reg_best.txt", "w") as f:
            f.write(f"Logistic Regression Performance: {log_reg_eval}")

        # with open(path_results_folder / f"{test_on}_svc_best.txt", "w") as f:
        #     f.write(f"Logistic Regression Performance: {svc_eval}")

        # with open(path_results_folder / f"{test_on}_rfc_best.txt", "w") as f:
        #     f.write(f"Logistic Regression Performance: {rfc_eval}")

        # with open(path_results_folder / f"{test_on}_gbc_best.txt", "w") as f:
        #     f.write(f"Logistic Regression Performance: {gbc_eval}")
