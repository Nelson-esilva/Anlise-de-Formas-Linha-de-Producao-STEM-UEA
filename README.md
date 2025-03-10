# Análise de Formas na Linha de Produção - STEM UEA

## Descrição

Este projeto visa desenvolver um sistema de inspeção automatizada para identificar diferentes formas geometricas em uma linha de produção. Utilizando técnicas de visão computacional e aprendizado de máquina, o sistema é capaz de detectar deformações e irregularidades nas latas, garantindo a qualidade do produto final.

## Estrutura do Projeto

A estrutura do projeto é organizada da seguinte forma:

```
Anlise-de-Formas-Linha-de-Producao-STEM-UEA/
├── datasets/
│   └── (Contém os conjuntos de dados utilizados para treinamento e teste)
├── projeto_latas/
│   └── (Arquivos relacionados ao desenvolvimento do projeto)
├── runs/
│   └── detect/
│       └── (Resultados das detecções realizadas)
├── .gitignore
├── LICENSE
├── README.md
├── trainV8.py
├── yolov8n.pt
```

- **datasets/**: Diretório que contém os conjuntos de dados utilizados para o treinamento e teste do modelo.
- **projeto_latas/**: Contém os arquivos e scripts relacionados ao desenvolvimento do projeto.
- **runs/detect/**: Armazena os resultados das detecções realizadas pelo modelo.
- **trainV8.py**: Script utilizado para treinar o modelo YOLOv8 com os dados disponíveis.
- **yolov8n.pt**: Arquivo do modelo pré-treinado YOLOv8.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação principal utilizada no desenvolvimento dos scripts.
- **YOLOv8**: Modelo de detecção de objetos utilizado para identificar defeitos nas latas.
- **OpenCV**: Biblioteca de visão computacional utilizada para processamento de imagens e vídeos.

## Como Executar o Projeto

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/Nelson-esilva/Anlise-de-Formas-Linha-de-Producao-STEM-UEA.git
   ```

2. **Instale as dependências necessárias:**

   Certifique-se de que o Python e o pip estão instalados. Em seguida, instale as bibliotecas requeridas:

   ```bash
   pip install -r requirements.txt
   ```

3. **Modifique o caminho para o modelo treinado no código:**

   No arquivo principal onde é feita a inferência (ex.: `main.py`), modifique a variável `model_path` para apontar para o caminho correto do modelo treinado (`best.pt`).

4. **Adicione o vídeo de exemplo:**

   Salve o vídeo que deseja processar na pasta `videos/` e atualize o caminho `video_path` no código.

5. **Realize a detecção de defeitos:**

   Utilize o script principal para processar o modelo de detecção sobre o vídeo adicionado anteriormente:

   ```bash
   python main.py
   ```

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para aprimorar o projeto.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.


