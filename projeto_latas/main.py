import torch
import cv2

# Carrega o modelo YOLOv5 pré-treinado (ou treinado com dados de latas)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Substitua 'best.pt' pelo seu modelo treinado

def detect_cans(frame):
    """Detecta latas em um frame usando YOLOv5"""
    
    # Redimensiona o frame para facilitar o processamento
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Faz a detecção com YOLO
    results = model(frame_resized)
    
    # Extrai as informações das detecções
    detections = results.xyxy[0].numpy()  # Coordenadas da caixa delimitadora
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det  # Coordenadas, confiança, e classe
        if conf > 0.5:  # Confiança mínima para considerar a detecção
            label = f"Lata: {conf:.2f}"
            cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_resized, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame_resized

# Processa um vídeo
video_path = "videos/lata_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Aplica a função de detecção de latas
    processed_frame = detect_cans(frame)

    # Exibe o resultado
    cv2.imshow("Detecção de Latas com YOLOv5", processed_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()