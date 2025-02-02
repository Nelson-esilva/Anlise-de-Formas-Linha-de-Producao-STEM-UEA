import torch
import cv2
import numpy as np

# Carrega o modelo YOLOv5 treinado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/Projeto Linha de Produção/Software_Deteccao/An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA/yolov5/runs/train/exp4/weights/best.pt')

def detect_in_image(image_path):
    """Detecta latas em uma imagem usando YOLOv5"""
    # Carrega a imagem
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return

    # Faz a detecção com YOLOv5
    results = model(frame)

    # Extrai as detecções
    detections = results.xyxy[0].cpu().numpy()  # (x1, y1, x2, y2, conf, cls)

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf > 0.1:  # Confiança mínima
            label = f"Lata: {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibe a imagem processada
    cv2.imshow("Detecção de Latas", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Caminho da imagem a ser processada
image_path = "C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/Projeto Linha de Produção/Software_Deteccao/An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA/datasets/train/images/teste3.jpg"
detect_in_image(image_path)
