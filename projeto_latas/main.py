import torch
import cv2
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
    
    # Itera sobre os resultados
    for result in results:
        # Extrai as caixas delimitadoras, confianças e classes
        boxes = result.boxes.xyxy.cpu().numpy()  # Caixas no formato (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confianças das detecções
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs das classes
        
        # Itera sobre as detecções
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:
                x1, y1, x2, y2 = map(int, box)  # Converte coordenadas para inteiros
                
                # Verifica se a classe está dentro do range válido
                if cls_id < len(class_names):
                    label = f"{class_names[cls_id]}: {conf:.2f}"
                else:
                    label = f"Desconhecido: {conf:.2f}"
            
                # Desenha a caixa delimitadora e o rótulo na imagem
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Caminho da imagem para teste
"""image_path = "./videos/teste.jpg"

# Carregar imagem
frame = cv2.imread(image_path)

if frame is None:
    print(f"Erro ao carregar a imagem: {image_path}")
    exit()

# Aplica a função de detecção
processed_frame = detect_objects(frame)

# Exibe o resultado
cv2.imshow("Detecção com YOLOv8", processed_frame)

# Aguarda uma tecla para fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# Caminho do vídeo para teste
video_path = "./videos/teste5.mp4"

# Captura de vídeo
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Aplica a função de detecção
    processed_frame = detect_objects(frame)
    
    # Exibe o resultado
    cv2.imshow("Detecção com YOLOv8", processed_frame)
    
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()