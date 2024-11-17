import torch
import clip
from PIL import Image
import requests

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Procesar texto
text_inputs = ["Recto", "Regular", "Slim", "Oversize"]

text_inputs = [x + "Silhouette Type" for x in text_inputs]
text = clip.tokenize(text_inputs).to(device)

# Procesar imagen
image_path = "../images/images/82_1107309_87084771-TM_B.jpg"  # Ajusta el nombre del archivo según las imágenes que tengas
image = Image.open(image_path)
image = preprocess(image).unsqueeze(0).to(device)

# Realizar inferencia
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

print('start')
# Calcular similitud
similarity = torch.nn.functional.cosine_similarity(text_features, image_features)
print(similarity)