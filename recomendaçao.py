import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image

# Função para carregar e pré-processar imagens
def extract_features(image_path, model):
    img = load_img(image_path, target_size=(224, 224))  
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 
    features = model.predict(img_array) 
    return features.flatten()

# Função para processar todas as imagens em um diretório
def process_image_dataset(image_dir, model):
    features = []
    image_paths = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            filepath = os.path.join(image_dir, filename)
            image_paths.append(filepath)
            features.append(extract_features(filepath, model))
    
    return np.array(features), image_paths

def recommend_images(input_image_path, database_features, database_image_paths, model, top_n=4):
    input_features = extract_features(input_image_path, model)
    similarities = cosine_similarity([input_features], database_features)
    sorted_indices = np.argsort(similarities[0])[::-1]  
    recommended_images = [database_image_paths[i] for i in sorted_indices[:top_n]]  
    return recommended_images

def display_images(image_paths, input_image_path):
    plt.figure(figsize=(15, 5))

    
    input_img = Image.open(input_image_path)
    plt.subplot(1, len(image_paths) + 1, 1)
    plt.imshow(input_img)
    plt.title("Input Image")
    plt.axis('off')

    
    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path)
        plt.subplot(1, len(image_paths) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Recommended {i + 1}")
        plt.axis('off')

    plt.show()

image_dataset_dir = "dataset/"  # Diretório com o dataset de imagens
input_image_path = "input.jpg"  # Caminho da imagem de entrada

# Carregar modelo pré-treinado
print("Carregando modelo pré-treinado...")
model = ResNet50(weights="imagenet", include_top=False, pooling='avg')


print("Processando o dataset de imagens...")
db_features, db_image_paths = process_image_dataset(image_dataset_dir, model)

print("Recomendando imagens similares...")
recommended = recommend_images(input_image_path, db_features, db_image_paths, model)

s
display_images(recommended, input_image_path)
