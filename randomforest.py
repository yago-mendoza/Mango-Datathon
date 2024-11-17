import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from xgboost import XGBClassifier


# Load merged dataset
data_path = "data/unique_products_with_attributes.csv" 
data = pd.read_csv(data_path)

# data.head()
# data.isnull().sum()

# Plot frecuencies of the categoric variables
cat_columns = ['des_color', 'des_sex', 'des_age', 'des_line', 
               'des_fabric', 'des_product_category', 'des_product_aggregated_family',
               'des_product_family', 'des_product_type']

for col in cat_columns:
    if col == 'des_color' or col=='des_product_type':
        print(f"\nFrecuencias para la columna '{col}':\n")
        # Crear tabla de frecuencias
        tabla_frecuencias = data[col].value_counts().reset_index()
        tabla_frecuencias.columns = [col, 'Frecuencia']  # Renombrar columnas
        print(tabla_frecuencias)
    
    else:
        print(f"\nFrecuencias de {col}:")
        print(data[col].value_counts())

        # Visualización de frecuencias
        plt.figure(figsize=(10, 5))
        sns.countplot(data=data, y=col, order=data[col].value_counts().index, palette='viridis')
        plt.title(f"Distribución de {col}")
        # plt.show() # uncomment to see the plots

# data.describe(include='all')

# Unique predictive cols
target_col = 'des_value'
predictive_cols = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
                   'knit_structure', 'heel_shape_type', 'length_type',
                   'sleeve_length_type', 'toecap_type', 'waist_type',
                   'closure_placement', 'cane_height_type']

for col in predictive_cols:
    unique_values = data[col].nunique()
    print(f"Cardinalidad de {col}: {unique_values}")

# data = data.dropna()

# Select all the features - 0.79
# features = ['des_color', 'des_sex', 'des_age', 'des_line', 'des_fabric', 'des_product_category', 'des_product_aggregated_family','des_product_family', 'des_product_type']

# selecte most important features - precison 0.8
features = ["des_sex", "des_age", "des_line", "des_fabric", "des_product_family", "des_product_type"]

unique_attributes = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
       'knit_structure', 'heel_shape_type', 'length_type',
       'sleeve_length_type', 'toecap_type', 'waist_type',
       'closure_placement', 'cane_height_type']

# coding the categories for each categorical colum
mappings = {}
for col in features + unique_attributes:
    original_data = data[col]
    # print(original_data.value_counts())
    encoded_data = data[col].astype("category").cat.codes
    # print(encoded_data.value_counts())
    data[col] = encoded_data
    
    df_i = pd.DataFrame({
        'original': original_data.value_counts().index,
        'encoding': encoded_data.value_counts().index
        })
    print(col,df_i)
    mappings[col] = df_i
mappings
with open(f"pretrained_models/mappings.pkl","wb") as f:
        pickle.dump(mappings,f)

# Split test and train models 
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Diccionario para almacenar modelos entrenados y sus resultados
models = {}
results = []
all_y_true = []  # Lista para almacenar todas las verdaderas etiquetas
all_y_pred = []  # Lista para almacenar todas las predicciones

# Entrenar un modelo Random Forest para cada attribute_name
for attribute in unique_attributes:
    print(f"Entrenando modelo para el atributo: {attribute}")

    # Separar características y etiquetas
    X_train = train_data[features]
    y_train = train_data[attribute]
    X_test = test_data[features]
    y_test = test_data[attribute]
    
    # Entrenar el modelo
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Realizar predicciones
    y_pred = rf.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión para {attribute}: {accuracy:.4f}")

    # Almacenar resultados y etiquetas
    models[attribute] = rf
    # save
    with open(f"pretrained_models/model_{attribute}.pkl","wb") as f:
        pickle.dump(rf,f)
    results.append(accuracy)
    all_y_true.extend(y_test)  # Guardamos las etiquetas verdaderas
    all_y_pred.extend(y_pred)  # Guardamos las predicciones

    # Map back to labels
    mapping_df = mappings[attribute]
    encoding_to_label = dict(zip(mapping_df['encoding'], mapping_df['original']))
    y_pred_labeled = [encoding_to_label[encoding] for encoding in y_pred]
    test_data[f"{attribute}_predicted"] = y_pred_labeled
    

# Calcular la precisión total (en función de todas las predicciones y etiquetas)
total_accuracy = accuracy_score(all_y_true, all_y_pred)
print(f"\nPrecisión total del modelo: {total_accuracy:.4f}")

test_data[["length_type","length_type_predicted"]]

# test_data["woven_structure"].value_counts()
# test_data["woven_structure_predicted"].value_counts()
# test_data.columns

data_path = "data/test_data.csv"
real_test_df = pd.read_csv(data_path)
real_test_df.head()

def map_to_encoding(mappings,value):
    # print("value=",value,len(value))
    # print("mappings=",mappings)
    # a = input("mfkdl")
    if value in mappings:
        # print("return=",mappings[value])
        return mappings[value]
    else:
        # print("not found")
        return 999
    
for feature in features:
    # Encode using mappings
    mapping_df = mappings[feature]
    encoding_to_label = dict(zip(mapping_df['original'],mapping_df['encoding']))
    print(encoding_to_label)
    # y_pred_labeled = [encoding_to_label[encoding] for encoding in y_pred]
    # test_data[f"{attribute}_predicted"] = y_pred_labeled
    
    real_test_df[feature] = real_test_df[feature].apply(lambda x: map_to_encoding(encoding_to_label,x))                                                        


# Create a prediction column with noting
real_test_df["des_value"] = ["" for _ in range(len(real_test_df))]

for attribute in unique_attributes:
    # Columns
    x = real_test_df[features]
    
    # Rows
    x = x[real_test_df["attribute_name"]==attribute]
    
    # Predict
    rf = models[attribute]
    y = rf.predict(x)
    # print(y)

    # Back to labels
    mapping_df = mappings[attribute]
    encoding_to_label = dict(zip(mapping_df['encoding'], mapping_df['original']))

    # y_pred_labeled = [encoding_to_label[encoding] for encoding in y]
    y_pred_labeled = []
    for encoding in y:
        # print(attribute,encoding,type(encoding),encoding_to_label)
        val_i = encoding_to_label[int(encoding)]
        y_pred_labeled.append(val_i)

    indexes = real_test_df["attribute_name"]==attribute
    
    real_test_df.loc[indexes,"des_value"] = y_pred_labeled

real_test_df[["test_id","des_value"]].to_csv("test1.csv",index=False)