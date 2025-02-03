import os
import cv2
import numpy as np
import albumentations as A
from glob import glob
from shutil import copyfile

# 🔹 Diretórios de entrada e saída
input_folder = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/datasets/train/images"  # Pasta com imagens
labels_folder = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/datasets/train/labels"
output_folder_images = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/datasets/train/augmentation/images" 
output_folder_labels = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/datasets/train/augmentation/labels"


# 🔹 Criar pasta de saída se não existir
os.makedirs(output_folder_images, exist_ok=True)
os.makedirs(output_folder_labels, exist_ok=True)

# 🔹 Definição das transformações
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Espelhamento horizontal
    A.VerticalFlip(p=0.2),  # Espelhamento vertical
    A.Rotate(limit=30, p=0.7),  # Rotação entre -30° e +30°
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Ajuste de brilho e contraste
    A.GaussNoise(var_limit=(10, 50), p=0.4),  # Adiciona ruído gaussiano
    A.Blur(blur_limit=5, p=0.3),  # Adiciona desfoque
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=15, p=0.5),  # Pequenos deslocamentos e escalas
    A.CLAHE(clip_limit=2.0, p=0.3),  # Equalização do histograma para melhorar contraste
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),  # Ajuste de cores
])

# 🔹 Função para carregar o label de um arquivo .txt
def load_label(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        class_id, x_center, y_center, width, height = map(float, parts)
        boxes.append([class_id, x_center, y_center, width, height])
    return boxes

# 🔹 Função para salvar o label em formato .txt
def save_label(output_path, boxes):
    with open(output_path, 'w') as file:
        for box in boxes:
            file.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n")

# 🔹 Coletar todas as imagens (JPG e PNG)
image_paths = glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png"))

# 🔹 Processar todas as imagens da pasta de entrada
total_generated = 0
for img_path in image_paths:
    # 🟢 Carregar imagem
    image = cv2.imread(img_path)

    if image is None:
        print(f"⚠️ Erro ao carregar: {img_path}")
        continue  # Pular para próxima imagem

    # 🟢 Coletar o caminho do arquivo de label correspondente
    label_path = os.path.join(labels_folder, os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt'))

    if not os.path.exists(label_path):
        print(f"⚠️ Label não encontrado para: {img_path}")
        continue

    # 🟢 Carregar os labels
    boxes = load_label(label_path)

    # 🟢 Gerar 20 imagens aumentadas por imagem original
    for i in range(20):  # Gera 20 imagens aumentadas por imagem original
        # 🟢 Aplicar a transformação de data augmentation
        augmented = transform(image=image)
        augmented_image = augmented['image']

        # 🟢 Atualizar os rótulos após a transformação (a função `albumentations` mantém as coordenadas da caixa)
        transformed_boxes = boxes  # As caixas não mudam diretamente, mas você pode ajustar conforme a transformação

        # 🟢 Criar nome único para cada imagem gerada
        filename = os.path.basename(img_path).split(".")[0]
        output_image_path = os.path.join(output_folder_images, f"{filename}_aug_{i}.jpg")
        output_label_path = os.path.join(output_folder_labels, f"{filename}_aug_{i}.txt")

        # 🟢 Salvar a imagem e os labels
        cv2.imwrite(output_image_path, augmented_image)
        save_label(output_label_path, transformed_boxes)

        total_generated += 1

print(f"\n🎉 Data augmentation concluída! {total_generated} novas imagens geradas em: {output_folder_images}")
