from ultralytics import YOLO

# para marcar as imagens
# https://www.makesense.ai/

def main():
    # Load a model
    #model = YOLO("yolov8n.yaml")  # build a new model from scratch
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data=r"C:/Users/Nelso/OneDrive/Documentos/Projetos/STEM/projetoSTEM-producao/Software_Deteccao/Analise-de-Falhas-na-Linha-de-Producao-STEM-UEA/datasets/datasets.yaml", epochs=50, device='cpu')  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format
    # print("path", path)


if __name__ == '__main__':
    # freeze_support()
    main()