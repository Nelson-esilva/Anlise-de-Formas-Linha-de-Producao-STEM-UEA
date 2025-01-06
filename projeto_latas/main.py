import cv2
import os


image_path = "C:\Users\Nelso\OneDrive\Área de Trabalho\Projetos\STEM\Projeto Linha de Produção\Software_Deteccao\An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA\images\normal\D_NQ_NP_967829-MLU72252518800_102023-O.png"

import cv2
import numpy as np

def detect_dented_can(frame):
    """Processa o quadro (frame) do vídeo e detecta se a lata está amassada."""
    
    # Redimensiona o frame para facilitar o processamento
    frame_resized = cv2.resize(frame, (400, 400))
    
    # Converte para escala de cinza
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Aplica um desfoque para reduzir ruídos
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detecção de bordas com o algoritmo Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos na imagem
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenha os contornos na imagem original
    frame_with_contours = frame_resized.copy()
    cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)
    
    # Análise: Se o número de contornos for muito alto ou contornos complexos forem detectados
    if len(contours) > 20:
        cv2.putText(frame_with_contours, "Lata Amassada", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame_with_contours, "Lata Normal", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame_with_contours

# Caminho para o vídeo de exemplo
video_path = "videos/lata_video.mp4"

# Captura do vídeo
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Erro ao abrir o vídeo: {video_path}")
    exit()

while True:
    # Captura quadro a quadro
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro ao capturar o quadro.")
        break

    # Aplica a função de detecção
    processed_frame = detect_dented_can(frame)

    # Exibe o resultado
    cv2.imshow("Detecção de Latas no Vídeo", processed_frame)

    # Pressione 'q' para sair do loop
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
