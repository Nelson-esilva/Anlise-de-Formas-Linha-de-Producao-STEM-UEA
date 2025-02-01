import torch
import cv2

# Carrega o modelo YOLOv5 personalizado treinado com seus dados
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:/Users/Nelso/OneDrive/Área de Trabalho/Projetos/STEM/Projeto Linha de Produção/Software_Deteccao/An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA/yolov5/runs/train/exp4/weights/best.pt')

  # Substitua 'best.pt' pelo seu modelo treinado

def detect_cans(frame):
    """Detecta latas em um frame usando YOLOv5"""
    
    # Redimensiona o frame para o tamanho esperado pelo modelo (640x640 é o tamanho padrão para YOLOv5)
    frame_resized = cv2.resize(frame, (640, 640))
    
    # Converte a imagem para o formato adequado para o YOLOv5 (RGB)
    img = frame_resized[..., ::-1]  # Converte BGR para RGB
    
    # Faz a detecção com YOLOv5
    results = model(img)  # O modelo retorna um objeto com as detecções
    
    # Extrai as informações das detecções
    detections = results.xyxy[0].cpu().numpy()  # Coordenadas da caixa delimitadora no formato (x1, y1, x2, y2, conf, cls)
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det  # Coordenadas, confiança e classe
        print(f"Classe: {cls}, Confiança: {conf}")
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
video_path = "./videos/lata_video.mp4"  # Assume que a pasta 'videos' está no mesmo diretório do script

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
