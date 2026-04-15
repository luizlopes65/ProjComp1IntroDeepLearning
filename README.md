# Detector de Objetos YOLOv3

Projeto simples de detecção de objetos usando YOLOv3 com OpenCV.

## Funcionalidade

Este projeto utiliza o modelo YOLOv3 (You Only Look Once) para detectar objetos em imagens. O sistema:

- Carrega uma imagem de entrada
- Processa a imagem através da rede neural YOLOv3
- Detecta objetos e suas localizações
- Desenha caixas delimitadoras (bounding boxes) ao redor dos objetos detectados
- Salva e exibe o resultado

## Requisitos

- Python 3.8+
- Arquivos do modelo YOLOv3:
  - `weights/yolov3.weights`
  - `cfg/yolov3.cfg`
  - `data/coco.names`

## Como Executar

### Usando UV (recomendado)

```bash
uv run python main.py
```

### Usando Poetry

```bash
poetry run python main.py
```

## Estrutura do Projeto

```
.
├── main.py           # Script principal
├── helpers.py        # Funções auxiliares de processamento
├── images/           # Pasta com imagens de entrada/saída
├── weights/          # Pesos do modelo YOLOv3
├── cfg/              # Arquivo de configuração do YOLOv3
└── data/             # Nomes das classes COCO
```

## Saída

- A imagem processada é salva em `images/output.jpg`
- Uma janela matplotlib exibe o resultado com as detecções