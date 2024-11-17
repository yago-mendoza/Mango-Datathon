import os
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd

"""
Este archivo genera el CSV de embeddings para las imágenes, redondeando los valores a 3 decimales.
"""

# Configuración del dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando el dispositivo: {device}")

# Cargar el modelo CLIP
model, preprocess = clip.load("ViT-B/32", device=device)

# Directorio de imágenes
images_dir = "../images/images"

# Lista para almacenar los datos
data = []

# Obtener la lista de archivos en el directorio de imágenes
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

# Verificar si hay imágenes en el directorio
if not image_files:
    print("No se encontraron imágenes en el directorio especificado.")
    exit()

# Procesar cada imagen
for idx, image_file in enumerate(image_files):
    try:
        # Ruta completa de la imagen
        image_path = os.path.join(images_dir, image_file)
        
        # Extraer el nombre modificado (antes y después de la primera barra baja)
        parts = image_file.split('_', 2)  # Dividir en 3 partes máximo
        if len(parts) < 2:
            print(f"El nombre de archivo '{image_file}' no sigue el formato esperado. Se omitirá.")
            continue
        modified_name = f"{parts[0]}_{parts[1]}"
        
        # Abrir y preprocesar la imagen
        image = Image.open(image_path).convert("RGB")
        image_preprocessed = preprocess(image).unsqueeze(0).to(device)
        
        # Obtener el embedding de la imagen
        with torch.no_grad():
            image_features = model.encode_image(image_preprocessed)
        
        # Normalizar el embedding
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Convertir a NumPy
        embedding_numpy = image_features.cpu().numpy().flatten()
        
        # Crear un diccionario para la fila
        row = {"name": modified_name}
        # Añadir cada dimensión del embedding como una columna separada
        for dim_idx, value in enumerate(embedding_numpy):
            # Redondear a 3 decimales
            row[f"dim_{dim_idx+1}"] = round(value, 5)
        
        # Añadir la fila a los datos
        data.append(row)
        
        # Opcional: Imprimir el progreso cada 10 imágenes
        if (idx + 1) % 10 == 0 or (idx + 1) == len(image_files):
            print(f"Procesadas {idx + 1}/{len(image_files)} imágenes.")
    
    except Exception as e:
        print(f"Error al procesar la imagen '{image_file}': {e}")
        continue

# Crear un DataFrame de pandas
df = pd.DataFrame(data)

# Ver la estructura del DataFrame
print("Estructura del DataFrame:")
print(df.head())

# Guardar el DataFrame en un archivo CSV
csv_path = "image_embeddings.csv"
df.to_csv(csv_path, index=False)
print(f"El dataset ha sido guardado en '{csv_path}'")
