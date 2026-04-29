import random
import colorsys
import numpy as np
import torch
from PIL import Image


# ==========================================
# FUNÇÕES DE UTILIDADE E PRÉ-PROCESSAMENTO
# ==========================================

def read_classes(classes_path):
    """Lê os nomes das classes de um arquivo."""
    with open(classes_path) as f:
        return [c.strip() for c in f.readlines()]


def generate_colors(class_names):
    """Gera cores distintas para cada classe usando HSV."""
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)
    random.shuffle(colors)
    random.seed(None)
    return colors


def letterbox_image(image, size=(416, 416)):
    """Redimensiona a imagem mantendo a proporção e preenchendo com barras cinzas."""
    iw, ih = image.size
    w, h = size
    # Encontra a escala ideal sem distorcer
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    # Redimensiona mantendo a proporção
    image = image.resize((nw, nh), Image.BICUBIC)

    # Cria um fundo cinza neutro e cola a imagem no centro
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def reverter_escala_caixas(boxes, img_size, original_shape):
    """Remove o efeito do letterbox e reajusta as caixas para o tamanho original da imagem."""
    iw, ih = original_shape
    w, h = img_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    # Calcula o tamanho das barras cinzas no formato normalizado (0 a 1)
    dx = (w - nw) / 2.0 / w
    dy = (h - nh) / 2.0 / h
    scale_w = nw / w
    scale_h = nh / h

    # Remove as barras (Lembrando que o formato é [y_min, x_min, y_max, x_max])
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dy) / scale_h
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dx) / scale_w

    # Multiplica pelas dimensões exatas da imagem original
    boxes[:, [0, 2]] *= ih # Multiplica o eixo Y pela Altura
    boxes[:, [1, 3]] *= iw # Multiplica o eixo X pela Largura

    return boxes


def scale_boxes(boxes, image_shape):
    """Escala as caixas para as dimensões da imagem."""
    height, width = image_shape
    image_dims = torch.tensor([width, height, width, height], dtype=torch.float32, device=boxes.device)
    return boxes * image_dims


def preprocess_image(img_path, model_image_size=(416, 416)):
    """Pré-processa a imagem para entrada no modelo."""
    # 1. Carrega a imagem original
    image = Image.open(img_path).convert('RGB')

    # 2. Aplica o Letterbox (mantém proporção) em vez de distorcer
    boxed_image = letterbox_image(image, model_image_size)

    # 3. Converte para Tensor
    image_data = np.array(boxed_image, dtype='float32') / 255.0
    image_data = image_data[:, :, ::-1].copy() # RGB para BGR

    image_data = np.transpose(image_data, (2, 0, 1))
    image_data = torch.from_numpy(image_data).unsqueeze(0)

    # Retorna a imagem ORIGINAL (para desenho) e os dados quadrados (para a rede)
    return image, image_data


def manual_nms(boxes, scores, classes, iou_threshold):
    """
    Non-Maximum Suppression (NMS) Manual - IMPLEMENTAÇÃO NECESSÁRIA
    
    Esta função deve implementar o algoritmo de NMS manualmente.
    
    Args:
        boxes: Tensor de caixas delimitadoras [N, 4] no formato [y_min, x_min, y_max, x_max]
        scores: Tensor de scores de confiança [N]
        iou_threshold: Limiar de IoU para suprimir caixas sobrepostas
    
    Returns:
        keep: Tensor com os índices das caixas a manter
    
    TODO: Implementar o algoritmo de NMS:
    1. Ordenar as caixas por score (decrescente)
    2. Para cada caixa:
       - Adicionar ao resultado se não sobrepõe muito com caixas já selecionadas
       - Calcular IoU (Intersection over Union) com caixas restantes
       - Remover caixas com IoU > iou_threshold
    3. Retornar índices das caixas mantidas
    """
    keep = []

    # PLACEHOLDER - Implementação necessária
    # Por enquanto, retorna todos os índices ordenados por score
    order = scores.argsort(descending=True)
    while (len(order) > 0):
        idx = order[0]
        keep.append(idx.item())
        if (len(order) == 1):
            break
        order = order[1:]
        classe_campea = classes[idx]
        campea = boxes[idx]
        classe_restantes = classes[order]
        restantes = boxes[order]

        xx1 = torch.max(campea[0], restantes[:, 0])
        yy1 = torch.max(campea[1], restantes[:, 1])
        xx2 = torch.min(campea[2], restantes[:, 2])
        yy2 = torch.min(campea[3], restantes[:, 3])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        area_intersecao = w * h
        area_campea = (campea[2] - campea[0]) * (campea[3] - campea[1])
        area_restantes = (restantes[:, 2] - restantes[:, 0]) * (restantes[:, 3] - restantes[:, 1])
        uniao = area_campea + area_restantes - area_intersecao
        iou = area_intersecao / uniao
        mask = (iou <= iou_threshold) | (classe_campea != classe_restantes)
        order = order[mask]

    return torch.tensor(keep, dtype=torch.long)

 
