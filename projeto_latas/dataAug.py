import cv2
import numpy as np
import albumentations as A
import os
from glob import glob
from PIL import Image

# 🔹 Diretórios de entrada e saída
input_folder = "C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/datasets/train/images"
output_folder = "C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/datasets/train/augmentation"

# 🔹 Criar pasta de saída se não existir
os.makedirs(output_folder, exist_ok=True)

# 🔹 Definição das transformações
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Espelhamento horizontal
    A.VerticalFlip(p=0.2),  # Espelhamento vertical
    A.Rotate(limit=30, p=0.7),  # Rotação entre -30° e +30°
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Ajuste de brilho e contraste
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),  # 🔹 Corrigido para ponto flutuante
    A.Blur(blur_limit=5, p=0.3),  # Adiciona desfoque
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=15, p=0.5),  # Pequenos deslocamentos e escalas
    A.CLAHE(clip_limit=2.0, p=0.3),  # Equalização do histograma para melhorar contraste
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Ajuste de cores
])

# 🔹 Coletar todas as imagens (JPG, JPEG, PNG)
image_paths = glob(os.path.join(input_folder, "*.jpg")) + \
              glob(os.path.join(input_folder, "*.jpeg")) + \
              glob(os.path.join(input_folder, "*.JPG")) + \
              glob(os.path.join(input_folder, "*.JPEG")) + \
              glob(os.path.join(input_folder, "*.png")) + \
              glob(os.path.join(input_folder, "*.PNG"))

# 🔹 Verifica se há imagens na pasta
total_images = len(image_paths)

if total_images == 0:
    print(f"⚠️ Nenhuma imagem encontrada em: {input_folder}")
    print("🔍 Verifique se a pasta contém imagens nas extensões: JPG, JPEG, PNG")
    exit()

print(f"📸 {total_images} imagens encontradas. Iniciando data augmentation...")

# 🔹 Processar todas as imagens da pasta de entrada
total_generated = 0

for img_path in image_paths:
    # 🟢 Tentativa de carregar imagem
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Erro ao carregar: {img_path}")
        continue  # Pular para próxima imagem

    # 🟢 Converte para RGB (OpenCV lê em BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    filename = os.path.basename(img_path).split(".")[0]  # Nome do arquivo sem extensão

    for i in range(20):  # Gera 20 imagens aumentadas
        augmented = transform(image=image)['image']  # Aplica transformações
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)  # Voltar para BGR

        # 🟢 Criar nome único para cada imagem gerada
        output_path = os.path.join(output_folder, f"{filename}_aug_{i}.jpg")
        cv2.imwrite(output_path, augmented)

        total_generated += 1

print(f"\n🎉 Data augmentation concluída! {total_generated} novas imagens geradas em: {output_folder}")
