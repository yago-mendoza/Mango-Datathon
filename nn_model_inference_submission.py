import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import re

# ============================================
# 1. Cargar el diccionario EMBEDDINGS
# ============================================

EMBEDDINGS_FILE = 'data/image_embeddings.csv'  # Ruta al archivo de embeddings
embeddings_loaded = False

if os.path.exists(EMBEDDINGS_FILE):
    try:
        df_embeddings = pd.read_csv(EMBEDDINGS_FILE)
        print(f"Archivo de embeddings '{EMBEDDINGS_FILE}' leído exitosamente.")
        print(f"Columnas en el archivo de embeddings: {df_embeddings.columns.tolist()}")

        # Update 'key_column' to 'name'
        key_column = 'name'  # Updated to match the actual column name
        if key_column not in df_embeddings.columns:
            print(f"Error: La columna clave '{key_column}' no se encontró en el archivo de embeddings.")
            print(f"Columnas disponibles: {df_embeddings.columns.tolist()}")
            exit(1)

        # Ensure 'name' and 'cod_modelo_color' are strings
        df_embeddings[key_column] = df_embeddings[key_column].astype(str)
        df_test['cod_modelo_color'] = df_test['cod_modelo_color'].astype(str)

        # Standardize formatting
        df_embeddings[key_column] = df_embeddings[key_column].str.strip().str.lower()
        df_test['cod_modelo_color'] = df_test['cod_modelo_color'].str.strip().str.lower()

        # Optionally remove non-alphanumeric characters
        df_embeddings[key_column] = df_embeddings[key_column].apply(lambda x: re.sub(r'\W+', '', x))
        df_test['cod_modelo_color'] = df_test['cod_modelo_color'].apply(lambda x: re.sub(r'\W+', '', x))

        # Convert DataFrame to dictionary
        EMBEDDINGS = {}
        for index, row in df_embeddings.iterrows():
            cod_modelo_color = row[key_column]
            embedding = row.drop(key_column).values.astype(float)
            EMBEDDINGS[cod_modelo_color] = embedding
        embeddings_loaded = True
        print("Diccionario EMBEDDINGS creado exitosamente desde CSV.")
    except Exception as e:
        print(f"Error al cargar el archivo de embeddings como CSV: {e}")
        exit(1)
else:
    print(f"Error: El archivo de embeddings '{EMBEDDINGS_FILE}' no se encontró.")
    exit(1)

# ============================================
# 2. Leer el archivo de prueba desde 'data/test_products.csv'
# ============================================

TEST_CSV_FILE = 'data/test_products.csv'  # Ruta al archivo de prueba

# Verificar si el archivo existe
if os.path.exists(TEST_CSV_FILE):
    df_test = pd.read_csv(TEST_CSV_FILE)
    print(f"Archivo de prueba '{TEST_CSV_FILE}' leído exitosamente.")
else:
    print(f"Error: El archivo de prueba '{TEST_CSV_FILE}' no se encontró.")
    exit(1)

# Verificar que las columnas necesarias existan
required_columns = ['cod_modelo_color', 'attribute_name', 'test_id']
missing_columns = [col for col in required_columns if col not in df_test.columns]
if missing_columns:
    print(f"Error: Las siguientes columnas faltan en el archivo de prueba: {missing_columns}")
    exit(1)

# Ensure 'cod_modelo_color' is standardized as above
df_test['cod_modelo_color'] = df_test['cod_modelo_color'].astype(str).str.strip().str.lower()
df_test['cod_modelo_color'] = df_test['cod_modelo_color'].apply(lambda x: re.sub(r'\W+', '', x))

# ============================================
# 3. Cargar modelos y LabelEncoders
# ============================================

MODEL_DIR = 'trained_models'  # Directorio donde están los modelos entrenados

# Obtener la lista de atributos únicos que necesitamos predecir
attributes_needed = df_test['attribute_name'].unique()

# Diccionarios para almacenar los modelos y LabelEncoders
models = {}
label_encoders = {}
attributes_to_remove = []

for attribute in attributes_needed:
    model_path = os.path.join(MODEL_DIR, f"model_{attribute}.h5")
    label_encoder_path = os.path.join(MODEL_DIR, f"label_encoder_{attribute}.pkl")

    # Verificar si el modelo y el LabelEncoder existen
    if os.path.exists(model_path) and os.path.exists(label_encoder_path):
        # Cargar el modelo
        model = load_model(model_path)
        models[attribute] = model

        # Cargar el LabelEncoder
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        label_encoders[attribute] = label_encoder
        print(f"Modelo y LabelEncoder para el atributo '{attribute}' cargados exitosamente.")
    else:
        print(f"Advertencia: El modelo o LabelEncoder para el atributo '{attribute}' no se encontró. Se omitirá este atributo.")
        attributes_to_remove.append(attribute)

# Eliminar los atributos sin modelos o LabelEncoders de df_test
if attributes_to_remove:
    df_test = df_test[~df_test['attribute_name'].isin(attributes_to_remove)]

# Verificar si hay modelos cargados
if not models:
    print("Error: No se cargó ningún modelo. Asegúrate de que los modelos y LabelEncoders existan en el directorio 'trained_models'.")
    exit(1)

# ============================================
# 4. Realizar predicciones
# ============================================

# Lista para almacenar los resultados
results = []

for index, row in df_test.iterrows():
    cod_modelo_color = row['cod_modelo_color']
    attribute_name = row['attribute_name']
    test_id = row['test_id']

    # Obtener el embedding correspondiente
    embedding = EMBEDDINGS.get(cod_modelo_color)

    if embedding is None:
        print(f"Advertencia: Embedding no encontrado para el producto '{cod_modelo_color}'. Se omitirá 'test_id' {test_id}.")
        continue  # Saltar si no hay embedding disponible

    # Obtener el modelo y LabelEncoder para el atributo actual
    model = models.get(attribute_name)
    label_encoder = label_encoders.get(attribute_name)

    if model is None or label_encoder is None:
        print(f"Advertencia: Modelo o LabelEncoder no disponible para el atributo '{attribute_name}'. Se omitirá 'test_id' {test_id}.")
        continue  # Saltar si no hay modelo o LabelEncoder disponible

    # Preparar el embedding para la predicción
    embedding_input = np.array(embedding).reshape(1, -1)

    # Realizar la predicción
    prediction = model.predict(embedding_input)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_label = label_encoder.inverse_transform(predicted_class)

    # Almacenar el resultado
    results.append({'test_id': test_id, 'attribute_value': predicted_label[0]})

# ============================================
# 5. Generar el archivo CSV de resultados
# ============================================

# Crear un DataFrame con los resultados
df_results = pd.DataFrame(results)

# Guardar en un archivo CSV
OUTPUT_CSV_FILE = 'submission.csv'  # Nombre del archivo de salida
df_results.to_csv(OUTPUT_CSV_FILE, index=False)

print(f"\nResultados guardados en '{OUTPUT_CSV_FILE}'.")
print(df_results.head())
