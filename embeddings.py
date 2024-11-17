import pandas as pd
import numpy as np

# ============================================
# 1. Definir las rutas a los archivos CSV
# ============================================

ATTRIBUTES_FILE_PATH = 'data/unique_products_with_attributes.csv'  # Ruta al archivo de atributos
EMBEDDINGS_FILE_PATH = 'data/image_embeddings.csv'              # Ruta al archivo de embeddings

# ============================================
# 2. Procesar unique_product_with_attributes.csv
# ============================================

# Nombre de la columna clave en unique_product_with_attributes.csv
ATTRIBUTES_KEY_COLUMN = 'cod_modelo_color'

try:
    # Leer el archivo CSV de atributos usando pandas
    df_attributes = pd.read_csv(ATTRIBUTES_FILE_PATH)
    print(f"Archivo '{ATTRIBUTES_FILE_PATH}' leído exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{ATTRIBUTES_FILE_PATH}' no se encontró.")
    exit(1)
except Exception as e:
    print(f"Error al leer '{ATTRIBUTES_FILE_PATH}': {e}")
    exit(1)

# Imprimir las columnas para verificar
print("\nColumnas en 'unique_product_with_attributes.csv':")
print(df_attributes.columns.tolist())

# Eliminar espacios adicionales en los nombres de las columnas
df_attributes.columns = df_attributes.columns.str.strip()

# Verificar nuevamente las columnas
print("\nColumnas después de eliminar espacios:")
print(df_attributes.columns.tolist())

# Verificar si la columna clave existe
if ATTRIBUTES_KEY_COLUMN not in df_attributes.columns:
    print(f"\nError: La columna '{ATTRIBUTES_KEY_COLUMN}' no se encuentra en '{ATTRIBUTES_FILE_PATH}'.")
    exit(1)

# Identificar las columnas de atributos (a partir de 'silhouette_type')
ATTRIBUTE_START_COLUMN = 'silhouette_type'

if ATTRIBUTE_START_COLUMN not in df_attributes.columns:
    print(f"\nError: La columna de inicio de atributos '{ATTRIBUTE_START_COLUMN}' no se encuentra en '{ATTRIBUTES_FILE_PATH}'.")
    exit(1)

# Seleccionar las columnas de atributos
attribute_columns = df_attributes.columns[df_attributes.columns.get_loc(ATTRIBUTE_START_COLUMN):]

print(f"\nColumnas de atributos seleccionadas (a partir de '{ATTRIBUTE_START_COLUMN}'):")
print(attribute_columns.tolist())

# Inicializar ATTR_DICTS: {attribute: {product: value}}
ATTR_DICTS = {attribute: {} for attribute in attribute_columns}

# Inicializar ATTRIBUTE_VALUES: {attribute: set(values)}
ATTRIBUTE_VALUES = {attribute: set() for attribute in attribute_columns}

# Poblar ATTR_DICTS y ATTRIBUTE_VALUES
for index, row in df_attributes.iterrows():
    product = row.get(ATTRIBUTES_KEY_COLUMN)
    if pd.isnull(product):
        print(f"Fila {index + 2}: '{ATTRIBUTES_KEY_COLUMN}' es nulo o no existe.")
        continue
    for attribute in attribute_columns:
        value = row[attribute]
        ATTR_DICTS[attribute][product] = value
        ATTRIBUTE_VALUES[attribute].add(value)

# Convertir los conjuntos en listas ordenadas
ATTRIBUTE_VALUES = {attribute: sorted(list(values)) for attribute, values in ATTRIBUTE_VALUES.items()}

# ============================================
# 3. Procesar image_embeddings.csv
# ============================================

try:
    # Leer el archivo CSV de embeddings usando pandas
    df_embeddings = pd.read_csv(EMBEDDINGS_FILE_PATH)
    print(f"\nArchivo '{EMBEDDINGS_FILE_PATH}' leído exitosamente.")
except FileNotFoundError:
    print(f"Error: El archivo '{EMBEDDINGS_FILE_PATH}' no se encontró.")
    exit(1)
except Exception as e:
    print(f"Error al leer '{EMBEDDINGS_FILE_PATH}': {e}")
    exit(1)

# Imprimir las columnas para verificar
print("\nColumnas en 'image_embeddings.csv':")
print(df_embeddings.columns.tolist())

# Eliminar espacios adicionales en los nombres de las columnas
df_embeddings.columns = df_embeddings.columns.str.strip()

# Verificar las columnas después de eliminar espacios
print("\nColumnas después de eliminar espacios:")
print(df_embeddings.columns.tolist())

# Identificar la columna clave en image_embeddings.csv
# Asumimos que la primera columna es el identificador del producto
EMBEDDINGS_KEY_COLUMN = df_embeddings.columns[0]
print(f"\nColumna clave en 'image_embeddings.csv': '{EMBEDDINGS_KEY_COLUMN}'")

# Inicializar EMBEDDINGS: {product: numpy array of embeddings}
EMBEDDINGS = {}

# Iterar sobre cada fila para poblar EMBEDDINGS
for index, row in df_embeddings.iterrows():
    product = row.get(EMBEDDINGS_KEY_COLUMN)
    if pd.isnull(product):
        print(f"Fila {index + 2}: '{EMBEDDINGS_KEY_COLUMN}' es nulo o no existe.")
        continue
    try:
        # Convertir las dimensiones a un array de floats
        vector = row.drop(EMBEDDINGS_KEY_COLUMN).values.astype(float)
        EMBEDDINGS[product] = vector
    except ValueError as e:
        print(f"Error al convertir embeddings para el producto '{product}' en la fila {index + 2}: {e}")

# ============================================
# 4. Verificar y Diferenciar las Estructuras
# ============================================

# Ejemplo de acceso a las estructuras generadas

# Acceder a un valor específico en ATTR_DICTS
PRODUCT_EXAMPLE = '83_1124642'  # Cambia esto según tus datos
ATTRIBUTE_EXAMPLE = 'des_color'  # Cambia esto según tus datos

color = ATTR_DICTS.get(ATTRIBUTE_EXAMPLE, {}).get(PRODUCT_EXAMPLE)
if color:
    print(f"\nEl color del producto '{PRODUCT_EXAMPLE}' es '{color}'.")
else:
    print(f"\nProducto '{PRODUCT_EXAMPLE}' o atributo '{ATTRIBUTE_EXAMPLE}' no encontrado.")

# Acceder a los valores únicos de un atributo en ATTRIBUTE_VALUES
ATTRIBUTE_VALUES_EXAMPLE = 'des_color'  # Cambia esto según tus datos
colors_available = ATTRIBUTE_VALUES.get(ATTRIBUTE_VALUES_EXAMPLE, [])
if colors_available:
    print(f"Colores disponibles para '{ATTRIBUTE_VALUES_EXAMPLE}': {colors_available}")
else:
    print(f"Atributo '{ATTRIBUTE_VALUES_EXAMPLE}' no encontrado.")

# Acceder al vector de embeddings de un producto en EMBEDDINGS
embedding = EMBEDDINGS.get(PRODUCT_EXAMPLE)
if embedding is not None:
    print(f"Embeddings del producto '{PRODUCT_EXAMPLE}': {embedding}")
else:
    print(f"Embeddings para el producto '{PRODUCT_EXAMPLE}' no encontrados.")

# =================

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ============================================
# 1. Asumimos que las siguientes estructuras están disponibles:
#    - ATTR_DICTS: {attribute: {product: value}}
#    - ATTRIBUTE_VALUES: {attribute: list of possible values}
#    - EMBEDDINGS: {product: numpy array of embeddings}
# ============================================

# Ejemplo: Si no las tienes en el entorno actual, puedes cargarlas desde archivos
# con pickle o recrearlas ejecutando el script anterior.

# ============================================
# 2. Entrenar una red neuronal para cada atributo
# ============================================

# Configurar el uso de la GPU si está disponible
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Directorio para guardar los modelos entrenados (opcional)
import os
MODEL_SAVE_DIR = 'trained_models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Recorrer cada atributo
for attribute in ATTRIBUTE_VALUES:
    print(f"\nEntrenando modelo para el atributo: '{attribute}'")
    
    # Obtener los productos que tienen embeddings y valor para el atributo actual
    products_with_embeddings = set(EMBEDDINGS.keys())
    products_with_attribute = set(ATTR_DICTS[attribute].keys())
    products_to_use = products_with_embeddings & products_with_attribute
    
    # Verificar si hay suficientes datos para entrenar
    if len(products_to_use) < 10:
        print(f"Advertencia: Solo hay {len(products_to_use)} productos disponibles para el atributo '{attribute}'. Se omite el entrenamiento.")
        continue
    
    # Preparar los datos de entrada (X) y etiquetas (y)
    X = np.array([EMBEDDINGS[product] for product in products_to_use])
    y = [ATTR_DICTS[attribute][product] for product in products_to_use]
    
    # Codificar las etiquetas categóricas
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    # Convertir etiquetas a formato categórico (one-hot encoding)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Dividir los datos en entrenamiento y validación
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    # Definir el modelo de la red neuronal (profunda)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Definir callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Entrenar el modelo
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluar el modelo en el conjunto de validación
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Precisión en validación para el atributo '{attribute}': {accuracy * 100:.2f}%")
    
    # Guardar el modelo entrenado (opcional)
    model_save_path = os.path.join(MODEL_SAVE_DIR, f"model_{attribute}.h5")
    model.save(model_save_path)
    print(f"Modelo para el atributo '{attribute}' guardado en '{model_save_path}'")
    
    # Guardar el LabelEncoder para uso futuro (opcional)
    import pickle
    label_encoder_path = os.path.join(MODEL_SAVE_DIR, f"label_encoder_{attribute}.pkl")
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"LabelEncoder para el atributo '{attribute}' guardado en '{label_encoder_path}'")
