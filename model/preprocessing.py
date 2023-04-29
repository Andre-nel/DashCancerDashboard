import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path
import pickle


def create_pipeline():

    pipeline = ImbPipeline([
        ("scaler", RobustScaler()),
        ("power_transformer", PowerTransformer()),
        ("smote", SMOTE(sampling_strategy="auto", random_state=42)),
        ("pca", PCA(n_components=0.95))
    ])

    return pipeline


def preprocess_cancer_data(data, pipeline, target=None, keep_all_features=True, already_fitted=False):

    if target is not None:
        data_scaled = pipeline.named_steps["scaler"].fit_transform(data)
        data_power_transformed = pipeline.named_steps["power_transformer"].fit_transform(data_scaled)
        data_resampled, target_resampled = pipeline.named_steps["smote"].fit_resample(data_power_transformed, target)
    else:
        data_scaled = pipeline.named_steps["scaler"].transform(data)
        data_power_transformed = pipeline.named_steps["power_transformer"].transform(data_scaled)
        data_resampled = data_power_transformed
        target_resampled = None

    if keep_all_features:
        return pd.DataFrame(data_resampled, columns=data.columns), target_resampled, pipeline

    data_pca = pipeline.named_steps["pca"].fit_transform(data_resampled)

    preprocessed_data = pd.DataFrame(data_pca, columns=[
        f'PC_{i}' for i in range(1, pipeline.named_steps['pca'].n_components_ + 1)
    ])

    return preprocessed_data, target_resampled, pipeline


if __name__ == "__main__":
    root_path = Path(__file__).parent.parent
    data_dir = root_path / "data"

    raw_data_path = data_dir / "BreastCancer.csv"
    pickled_path = root_path / "pickled"

    preprocessed_all_features_path = data_dir / "preprocessed_cancer_data_all_features.csv"
    preprocessed_pca_path = data_dir / "preprocessed_cancer_data_pca.csv"

    pipeline_fit_to_all_features_path = pickled_path / "preprocessor_pipeline_all_features.pkl"
    pipeline_fit_to_pca_path = pickled_path / "preprocessor_pipeline_pca.pkl"

    data = pd.read_csv(raw_data_path)
    data.drop(columns=["Unnamed: 32", "id"], inplace=True)

    target_data = data["diagnosis"].map({"M": 1, "B": 0})
    data['diagnosis_numeric'] = data["diagnosis"].map({"M": 1, "B": 0})
    feature_data = data.drop(columns=["diagnosis", "diagnosis_numeric"])
    target_data = data["diagnosis_numeric"]

    # Type of preprocessing to be done
    for keep_all_features in [True, False]:
        preprocessor_pipeline = create_pipeline()

        preprocessed_feature_data, diagnosis_resampled, preprocessor_pipeline = preprocess_cancer_data(
                                                                                    feature_data,
                                                                                    preprocessor_pipeline,
                                                                                    target_data,
                                                                                    keep_all_features=keep_all_features
                                                                                )

        if diagnosis_resampled is not None:
            diagnosis_resampled = pd.Series(diagnosis_resampled, name="diagnosis")
            preprocessed_data = pd.concat([preprocessed_feature_data, diagnosis_resampled], axis=1)
        else:
            preprocessed_data = preprocessed_feature_data

        # save the preprocessed Data and Trained pipelines
        if keep_all_features:
            preprocessed_cancer_data_all_features = preprocessed_data

            preprocessed_cancer_data_all_features.to_csv(preprocessed_all_features_path, index=False)
            with open(pipeline_fit_to_all_features_path, 'wb') as f:
                pickle.dump(preprocessor_pipeline, f)
        else:
            preprocessed_cancer_data_pca = preprocessed_data

            preprocessed_cancer_data_pca.to_csv(preprocessed_pca_path, index=False)
            with open(pipeline_fit_to_pca_path, 'wb') as f:
                pickle.dump(preprocessor_pipeline, f)
