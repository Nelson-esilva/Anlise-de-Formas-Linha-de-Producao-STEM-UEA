import torch
import cv2
import os
import json
from ultralytics import YOLO

# Caminho para o modelo treinado no YOLOv8
model_path = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/runs/detect/train5/weights/best.pt"

# Lista de nomes das classes (deve ser igual ao YAML)
class_names = ["cilindro", "cubo", "esfera"]

# Carregar o modelo treinado
model = YOLO(model_path)

print("Modelo carregado com sucesso:", model)

def detect_objects(frame):
    """Detecta objetos em um frame usando YOLOv8"""
    
    # Faz a detecção com YOLOv8
    results = model(frame)  # O modelo retorna uma lista de objetos Results
    
    # Lista para armazenar as detecções do frame atual
    frame_detections = []
    
    # Itera sobre os resultados
    for result in results:
        # Extrai as caixas delimitadoras, confianças e classes
        boxes = result.boxes.xyxy.cpu().numpy()  # Caixas no formato (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confianças das detecções
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs das classes
        
        # Itera sobre as detecções
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  # Filtra detecções com confiança maior que 50%
                x1, y1, x2, y2 = map(int, box)  # Converte coordenadas para inteiros
                
                # Verifica se a classe está dentro do range válido
                if cls_id < len(class_names):
                    label = class_names[cls_id]
                else:
                    label = "Desconhecido"
                
                # Adiciona a detecção à lista
                frame_detections.append({
                    'shape': label,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf)
                })
    
    return frame_detections

def process_folder(input_folder, output_folder):
    """Processa todas as imagens de uma pasta e salva as detecções em um arquivo JSON"""
    
    # Lista para armazenar os resultados de detecção
    detections = []
    
    # Verifica se a pasta de saída existe, caso contrário, cria
    os.makedirs(output_folder, exist_ok=True)
    
    # Itera sobre todas as imagens na pasta de entrada
    for frame_number, image_name in enumerate(sorted(os.listdir(input_folder))):
        if image_name.endswith('.png') or image_name.endswith('.jpg'):
            image_path = os.path.join(input_folder, image_name)
            
            # Carrega a imagem
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Erro ao carregar a imagem: {image_path}")
                continue
            
            # Faz a detecção com YOLOv8
            frame_detections = detect_objects(frame)
            
            # Adiciona as detecções do frame atual à lista geral
            for detection in frame_detections:
                detections.append({
                    'frame': frame_number,
                    'object_name': f'object_{frame_number}',
                    **detection  # Inclui shape, bbox e confidence
                })
    
    # Salva as detecções em um arquivo JSON
    output_path = os.path.join(output_folder, 'detections.json')
    with open(output_path, 'w') as f:
        json.dump(detections, f, indent=4)
    
    print(f"Detecções salvas em {output_path}")
    

# Caminho da pasta com as imagens renderizadas do Blender
input_folder = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/blender/images"

# Caminho da pasta de saída para salvar o JSON
output_folder = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/blender/json"

# Processa as imagens e salva as detecções
process_folder(input_folder, output_folder)