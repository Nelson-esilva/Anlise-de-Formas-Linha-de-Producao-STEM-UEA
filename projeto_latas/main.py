import cv2
import os


image_path = "C:\Users\Nelso\OneDrive\Área de Trabalho\Projetos\STEM\Projeto Linha de Produção\Software_Deteccao\An-lise-de-Falhas-na-Linha-de-Produ-o---STEM-UEA\images\normal\D_NQ_NP_967829-MLU72252518800_102023-O.png"

def process_image(image_path):
    
    # Processa a imagem e detecta se ela esta amassada.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return
    

    #Redimensiona a imagem para o processamento
    image = cv2.resize(image, (400,400))
    cv2.imshow("Original", image)

    #Converte para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Cinza", gray)
