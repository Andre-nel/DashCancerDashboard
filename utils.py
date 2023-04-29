import pickle
import numpy as np
import pandas as pd

from model.preprocessing import preprocess_cancer_data


def load_pickle(path):
    with open(path, 'rb') as file:
        pickle_file = pickle.load(file)
    return pickle_file


def predict_diagnosis(model, features, path_to_preprocessor_pipeline, target=None, keep_all_features=True):
    features_array = np.array(features).reshape(1, -1)

    preprocessor_pipeline = load_pickle(path=path_to_preprocessor_pipeline)

    # Use the original feature names when creating the DataFrame
    original_feature_names = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                              'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                              'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                              'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                              'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                              'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                              'symmetry_worst', 'fractal_dimension_worst']
    features_df = pd.DataFrame(features_array, columns=original_feature_names)

    preprocessed_feature_data, _, _ = preprocess_cancer_data(
      features_df, preprocessor_pipeline, target, keep_all_features)

    prediction_proba = model.predict_proba(preprocessed_feature_data)
    return prediction_proba
