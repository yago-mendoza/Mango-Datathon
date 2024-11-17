import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier



def perform_inference(row):
    unique_attributes = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
                         'knit_structure', 'heel_shape_type', 'length_type',
                         'sleeve_length_type', 'toecap_type', 'waist_type',
                         'closure_placement', 'cane_height_type']
    features = ["des_sex", "des_age", "des_line", "des_fabric", "des_product_family", "des_product_type"]
    
    # Load mappings
    with open(f"pretrained_models/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)

    # Categorize labels
    for feature_i in features:
        mapping_i = mappings[feature_i]
        encoding_to_label = dict(zip(mapping_i['original'], mapping_i['encoding']))
        row[feature_i] = [encoding_to_label[val] for val in row[feature_i]]

    # Select columns
    x = row[features]
    # Inference
    predictions = []
    for attribute in unique_attributes:
        with open(f"pretrained_models/model_{attribute}.pkl", "rb") as f:
            rf_i = pickle.load(f)
        y_i = rf_i.predict(x)

        # Back to labels
        mapping_i = mappings[attribute]
        encoding_to_label = dict(zip(mapping_i['encoding'], mapping_i['original']))
        predictions.append(encoding_to_label[y_i[0]])

    return pd.DataFrame([predictions],columns=unique_attributes)