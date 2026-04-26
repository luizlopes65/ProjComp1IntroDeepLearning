# YOLOv3 Object Detection - PyTorch Implementation

Implementação modular do YOLOv3 (You Only Look Once v3) em PyTorch para detecção de objetos em imagens.

## 📋 Descrição

Este projeto implementa o modelo YOLOv3 usando PyTorch puro, sem dependências de frameworks de alto nível. O código foi refatorado do notebook original em uma estrutura modular e organizada.

### Características

- ✅ Implementação completa do YOLOv3 com Darknet-53 backbone
- ✅ Suporte para 3 escalas de detecção (objetos pequenos, médios e grandes)
- ✅ Carregamento de pesos pré-treinados (.weights)
- ✅ Pré-processamento com letterbox (mantém proporção da imagem)
- ✅ Decodificação de predições YOLO
- ✅ Visualização de resultados com matplotlib
- ⚠️ NMS (Non-Maximum Suppression) manual - **REQUER IMPLEMENTAÇÃO**

## 🏗️ Estrutura do Projeto

```
.
├── config.py          # Constantes e âncoras do YOLOv3
├── utils.py           # Funções utilitárias (pré-processamento, NMS)
├── model.py           # Arquitetura da rede neural (ConvBlock, ResBlock, YOLOv3)
├── weights.py         # Carregamento de pesos binários .weights
├── inference.py       # Decodificação YOLO e pipeline de predição
├── main.py            # Script principal (plug-and-play)
├── pyproject.toml     # Dependências do projeto
├── coco.names         # Nomes das 80 classes COCO
├── yolov3.weights     # Pesos pré-treinados (baixar separadamente)
└── img/               # Diretório de imagens de teste
    ├── dog.jpg
    └── food.jpg
```

## 📦 Requisitos

- Python 3.12+
- PyTorch 2.0+
- Arquivos necessários:
  - `yolov3.weights` - Pesos pré-treinados do YOLOv3
  - `coco.names` - Lista com 80 classes do dataset COCO
  - Imagens de teste no diretório `img/`

## 🚀 Como Executar

### 1. Instalar Dependências

#### Usando UV (recomendado)
```bash
uv sync
```

#### Usando Poetry
```bash
poetry install
```

### 2. Baixar Pesos do Modelo

Baixe o arquivo `yolov3.weights` (aproximadamente 248 MB):
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```

### 3. Executar o Script Principal

#### Usando UV
```bash
uv run python main.py
```

#### Usando Poetry
```bash
poetry run python main.py
```

O script irá:
1. Carregar as classes do COCO
2. Instanciar o modelo YOLOv3
3. Converter os pesos .weights para .pth (primeira execução)
4. Executar predições nas imagens de teste
5. Exibir os resultados com matplotlib

## ⚠️ Implementação Necessária

### NMS Manual (Non-Maximum Suppression)

A função `manual_nms()` em `utils.py` está como **PLACEHOLDER** e precisa ser implementada:

```python
def manual_nms(boxes, scores, iou_threshold):
    """
    Implementar o algoritmo de NMS manualmente.
    
    Args:
        boxes: Tensor [N, 4] com caixas [y_min, x_min, y_max, x_max]
        scores: Tensor [N] com scores de confiança
        iou_threshold: Limiar de IoU para suprimir caixas
    
    Returns:
        keep: Tensor com índices das caixas a manter
    """
    # TODO: Implementar NMS
    pass
```

**Algoritmo sugerido:**
1. Ordenar caixas por score (decrescente)
2. Para cada caixa:
   - Adicionar ao resultado se não sobrepõe muito com caixas já selecionadas
   - Calcular IoU (Intersection over Union) com caixas restantes
   - Remover caixas com IoU > iou_threshold
3. Retornar índices das caixas mantidas

## 📚 Módulos

### `config.py`
Contém as âncoras do YOLOv3 para as 3 escalas de detecção.

### `utils.py`
Funções utilitárias:
- `read_classes()` - Carrega nomes das classes
- `generate_colors()` - Gera cores distintas para visualização
- `letterbox_image()` - Redimensiona mantendo proporção
- `reverter_escala_caixas()` - Remove padding do letterbox
- `preprocess_image()` - Pipeline completo de pré-processamento
- `manual_nms()` - **NMS manual (IMPLEMENTAR)**

### `model.py`
Arquitetura da rede:
- `ConvBlock` - Bloco convolucional básico
- `ResBlock` - Bloco residual
- `YOLOv3` - Modelo completo com Darknet-53 + FPN

### `weights.py`
- `carregar_pesos_yolov3()` - Carrega pesos do formato binário .weights

### `inference.py`
Pipeline de inferência:
- `decode_yolo()` - Decodifica saídas brutas do YOLO
- `executar_predicao()` - Pipeline completo de predição + visualização

### `main.py`
Script principal plug-and-play que orquestra todo o processo.

## 🎯 Parâmetros Configuráveis

No `main.py`, você pode ajustar:
- `score_threshold` - Limiar de confiança mínima (padrão: 0.5)
- `iou_threshold` - Limiar de IoU para NMS (padrão: 0.4)
- Caminhos das imagens de teste

## 📝 Notas

- O código preserva todos os comentários originais em português
- A primeira execução converte `.weights` para `.pth` (mais rápido nas próximas)
- O modelo detecta até 80 classes do dataset COCO
- Máximo de 10 detecções por imagem (configurável)

## 🔗 Referências

- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [Darknet Original](https://pjreddie.com/darknet/yolo/)
- [COCO Dataset](https://cocodataset.org/)

## 📄 Licença

Este projeto é para fins educacionais.