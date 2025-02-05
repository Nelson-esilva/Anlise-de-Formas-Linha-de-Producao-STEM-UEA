import torch
import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado no YOLOv8
model_path = r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/runs/detect/train2/weights/best.pt"

# Carregar o modelo treinado
model = YOLO(model_path)

# Verificar se o modelo foi carregado corretamente
print("Modelo carregado com sucesso:", model)


def detect_cans(frame):
    """Detecta latas em um frame usando YOLOv8"""
    
    # Redimensiona o frame para o tamanho esperado pelo modelo (640x640 é o tamanho padrão para YOLOv8)
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Converte a imagem para o formato adequado para o YOLOv8 (RGB)
    img = frame_resized[..., ::-1]  # Converte BGR para RGB
    
    # Faz a detecção com YOLOv8
    results = model(img)  # O modelo retorna uma lista de objetos Results
    
    # Itera sobre os resultados
    for result in results:
        # Extrai as caixas delimitadoras, confianças e classes
        boxes = result.boxes.xyxy.cpu().numpy()  # Caixas no formato (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confianças das detecções
        class_ids = result.boxes.cls.cpu().numpy()  # IDs das classes
        
        # Itera sobre as detecções
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box  # Coordenadas da caixa delimitadora
            print(f"Classe: {cls_id}, Confiança: {conf}")
            
            if conf > 0.1:  # Confiança mínima para considerar a detecção
                label = f"Lata: {conf:.2f}"
                
                # Redimensiona as coordenadas da caixa delimitadora para o tamanho original da imagem
                x1, y1, x2, y2 = int(x1 * frame.shape[1] / 640), int(y1 * frame.shape[0] / 640), \
                                 int(x2 * frame.shape[1] / 640), int(y2 * frame.shape[0] / 640)
                
                # Desenha a caixa delimitadora e o rótulo na imagem
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


# Processa um vídeo
video_path = "./videos/teste1.mp4"  # Assume que a pasta 'videos' está no mesmo diretório do script

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
    cv2.imshow("Detecção de Latas com YOLOv8", processed_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()