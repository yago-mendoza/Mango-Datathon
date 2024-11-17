import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle
import lightgbm as lgb
import warnings

# Ignorar advertencias para mantener la salida limpia
warnings.filterwarnings('ignore')

# Crear carpetas si no existen
os.makedirs("pretrained_models", exist_ok=True)

# Cargar el dataset
data_path = "data/unique_products_with_attributes.csv" 
data = pd.read_csv(data_path)

# Selección de características y atributos únicos
features = ["des_sex", "des_age", "des_line", "des_fabric", "des_product_family", "des_product_type"]
unique_attributes = ['silhouette_type', 'neck_lapel_type', 'woven_structure',
                     'knit_structure', 'heel_shape_type', 'length_type',
                     'sleeve_length_type', 'toecap_type', 'waist_type',
                     'closure_placement', 'cane_height_type']

# Codificación de variables categóricas y almacenamiento de mappings
mappings = {}
for col in features + unique_attributes:
    data[col] = data[col].astype('category')
    data[col], mapping = data[col].cat.codes, dict(enumerate(data[col].cat.categories))
    mappings[col] = mapping  # mapping: {code: category}

# Guardar los mappings
with open("pretrained_models/mappings.pkl", "wb") as f:
    pickle.dump(mappings, f)

# Manejo de valores nulos
data = data.fillna(-1)

# División de datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

# Diccionarios para almacenar modelos y resultados
models = {}
results = []
all_y_true = []
all_y_pred = []

# Entrenamiento de un modelo LightGBM para cada atributo
for attribute in unique_attributes:
    print(f"\nEntrenando modelo para el atributo: {attribute}")
    
    # Separar características y etiquetas
    X_train = train_data[features]
    y_train = train_data[attribute]
    X_test = test_data[features]
    y_test = test_data[attribute]
    
    # Definir el modelo
    model = lgb.LGBMClassifier(random_state=42)
    
    # Definir el espacio de hiperparámetros para GridSearch
    param_grid = {
        'num_leaves': [31, 50],
        'n_estimators': [100, 200],
        'learning_rate': [0.1, 0.01]
    }
    
    # Configurar Grid Search
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    
    # Entrenar el modelo
    grid_search.fit(X_train, y_train)
    
    # Obtener el mejor modelo
    best_model = grid_search.best_estimator_
    
    # Realizar predicciones en el conjunto de prueba
    y_pred = best_model.predict(X_test)
    
    # Calcular la precisión
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Mejor Precisión para {attribute}: {accuracy:.4f}")
    print(f"Mejores hiperparámetros para {attribute}: {grid_search.best_params_}")
    
    # Almacenar el modelo
    models[attribute] = best_model
    with open(f"pretrained_models/model_{attribute}.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Almacenar los resultados
    results.append(accuracy)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Calcular la precisión total
total_accuracy = accuracy_score(all_y_true, all_y_pred)
print(f"\nPrecisión total del modelo: {total_accuracy:.4f}")

# Cargar el conjunto de prueba real
real_test_path = "data/test_data.csv"
real_test_df = pd.read_csv(real_test_path)

# Codificar las características del conjunto de prueba real usando los mappings
for feature in features:
    mapping = mappings[feature]  # mapping: {code: category}
    inv_mapping = {v: k for k, v in mapping.items()}  # inv_mapping: {category: code}
    # Mapear las categorías a códigos; las desconocidas se asignan a -1
    real_test_df[feature] = real_test_df[feature].map(inv_mapping).fillna(-1).astype(int)

# Manejo de valores nulos en el conjunto de prueba real
real_test_df = real_test_df.fillna(-1)

# Crear una columna 'des_value' vacía
real_test_df["des_value"] = ""

# Realizar predicciones para cada atributo
for attribute in unique_attributes:
    print(f"\nPrediciendo para el atributo: {attribute}")
    
    # Filtrar las filas que corresponden al atributo actual
    attribute_mask = real_test_df["attribute_name"] == attribute
    X_attr = real_test_df.loc[attribute_mask, features]
    
    if X_attr.empty:
        print(f"No hay datos para el atributo {attribute} en el conjunto de prueba.")
        continue
    
    # Realizar la predicción
    model = models[attribute]
    y_pred_attr = model.predict(X_attr)
    
    # Convertir las predicciones a las etiquetas originales
    mapping = mappings[attribute]  # mapping: {code: category}
    y_pred_labels = [mapping.get(code, "unknown") for code in y_pred_attr]
    
    # Asignar las predicciones al DataFrame
    real_test_df.loc[attribute_mask, "des_value"] = y_pred_labels

# Guardar las predicciones en un archivo CSV
output_path = "test1.csv"
real_test_df[["test_id", "des_value"]].to_csv(output_path, index=False)
print(f"\nPredicciones guardadas en {output_path}")
