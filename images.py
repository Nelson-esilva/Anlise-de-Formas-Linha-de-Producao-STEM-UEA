import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Configurações
query = "cilindro"  # Termo de busca
num_images = 100  # Número de imagens para baixar
save_folder = r"C:/Users/Nelso/OneDrive/Imagens/latas"  # Pasta para salvar as imagens

# Cria a pasta de destino
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
                     
# URL de busca (exemplo: Google Imagens)
url = f"https://www.istockphoto.com/br/search/more-like-this/839499726?assettype=image&page=3"

# Headers para simular um navegador
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Faz a requisição
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Encontra todas as tags de imagem
images = soup.find_all("img")

# Baixa as imagens
for i, img in enumerate(images[:num_images]):
    img_url = img["src"]
    if not img_url.startswith("http"):
        img_url = urljoin(url, img_url)
    
    try:
        img_data = requests.get(img_url).content
        with open(os.path.join(save_folder, f"can_{i+1}.jpg"), "wb") as f:
            f.write(img_data)
        print(f"Imagem {i+1} baixada com sucesso!")
    except Exception as e:
        print(f"Erro ao baixar a imagem {i+1}: {e}")