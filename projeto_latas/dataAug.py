import cv2
import numpy as np
import albumentations as A
import os
from glob import glob

# Diretórios de entrada e saída
input_folder = r"C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/Projeto Linha de Produção/Software_Deteccao/An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA/datasets/train/images"
output_folder = r"C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/Projeto Linha de Produção/Software_Deteccao/An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA/datasets/train/augmentation"

# Cria a pasta de saída se não existir
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Definição das transformações
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Espelhamento horizontal
    A.Rotate(limit=20, p=0.5),  # Rotação entre -20° e +20°
    A.RandomBrightnessContrast(p=0.5),  # Ajusta brilho e contraste
    A.GaussNoise(var_limit=(10, 50), p=0.5),  # Correção: adicionada a vírgula no final da linha
    A.Blur(blur_limit=3, p=0.2),  # Adiciona desfoque
    A.Affine(translate_percent=0.05, scale=0.95, rotate=15, p=0.5)  # Corrigido ShiftScaleRotate
])

# Processa todas as imagens da pasta de entrada
image_paths = glob(os.path.join(input_folder, "*.jpg"))  # Ajuste a extensão se necessário

for img_path in image_paths:
    image = cv2.imread(img_path)  # Lê a imagem

    if image is None:  # Verifica se a imagem foi carregada corretamente
        print(f"Erro ao carregar imagem: {img_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte para RGB

    for i in range(5):  # Gera 5 imagens novas por original
        augmented = transform(image=image)['image']  # Aplica transformação
        augmented = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)  # Converte de volta para BGR
        filename = os.path.basename(img_path).split(".")[0]  # Nome do arquivo sem extensão
        cv2.imwrite(os.path.join(output_folder, f"{filename}_aug_{i}.jpg"), augmented)  # Salva a imagem

print(f"Data augmentation concluída! Imagens salvas em {output_folder}")
