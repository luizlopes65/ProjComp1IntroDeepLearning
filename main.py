import torch
from config import YOLOV3_ANCHORS
from utils import read_classes
from model import YOLOv3
from weights import carregar_pesos_yolov3
from inference import executar_predicao


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

def main():
    # Configuração do dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processamento via: {device}")

    # 1. Carregue as classes
    class_names = read_classes("data/coco.names")  # Certifique-se de que o arquivo tenha 80 linhas
    print(f"Carregadas {len(class_names)} classes")

    # 2. Instancie o YOLOv3
    modelo = YOLOv3(num_classes=len(class_names)).to(device)
    print("Modelo YOLOv3 instanciado")

    # 3. Converter e Salvar os Pesos (Rode isso a primeira vez)
    try:
        arquivo_weights = "weights/yolov3.weights"
        modelo = carregar_pesos_yolov3(arquivo_weights, modelo)
        torch.save(modelo.state_dict(), "yolov3_convertido.pth")
        print("Pesos convertidos e salvos em yolov3_convertido.pth")
    except Exception as e:
        print(f"Aviso na conversão (talvez os pesos já existam ou o arquivo não foi encontrado): {e}")


    try:
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
        print("Pesos carregados de yolov3_convertido.pth")
    except Exception as e:
        print(f"Erro ao carregar pesos: {e}")
        return

    # 5. Execute predições nas imagens de teste
    print("\n" + "="*50)
    print("Executando predições...")
    print("="*50 + "\n")
    
    # Coloque o nome da sua imagem de teste aqui
    executar_predicao("images/dog.jpg", modelo, class_names, YOLOV3_ANCHORS, device, score_threshold=0.5)
    executar_predicao("images/food.jpg", modelo, class_names, YOLOV3_ANCHORS, device)


if __name__ == "__main__":
    main()

 
