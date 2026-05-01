import numpy as np
import torch
import torch.nn as nn
from model import ConvBlock
from ultralytics import YOLO
from pathlib import Path


# ==========================================
# CARREGAR PESOS (EXTRATOR BINÁRIO)
# ==========================================

def carregar_pesos_yolov3(caminho_weights, modelo):
    """ Extrai pesos .weights (C) injetando na estrutura do PyTorch """
    print(f"Iniciando leitura de: {caminho_weights}")
    with open(caminho_weights, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    # Identificamos ordenadamente todos os blocos e convoluções lineares
    modulos = []
    for m in modelo.modules():
        if isinstance(m, ConvBlock) or (isinstance(m, nn.Conv2d) and m.bias is not None):
            modulos.append(m)

    ptr = 0
    for i, modulo in enumerate(modulos):
        if isinstance(modulo, ConvBlock):
            conv, bn = modulo.conv, modulo.bn

            # Carrega BatchNorm
            num_bn = bn.bias.numel()

            bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.bias))
            ptr += num_bn

            bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.weight))
            ptr += num_bn

            bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_mean))
            ptr += num_bn

            bn_var = torch.from_numpy(weights[ptr:ptr+num_bn]).view_as(bn.running_var)
            bn.running_var.data.copy_(torch.clamp(bn_var, min=1e-5)) # Escudo Anti-NaN
            ptr += num_bn

            # Carrega Convolução do Bloco
            num_w = conv.weight.numel()
            conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(conv.weight))
            ptr += num_w

        elif isinstance(modulo, nn.Conv2d):
            # Camada Linear Final do YOLO (sem BN, com Bias)
            num_b = modulo.bias.numel()
            modulo.bias.data.copy_(torch.from_numpy(weights[ptr:ptr+num_b]).view_as(modulo.bias))
            ptr += num_b

            num_w = modulo.weight.numel()
            modulo.weight.data.copy_(torch.from_numpy(weights[ptr:ptr+num_w]).view_as(modulo.weight))
            ptr += num_w

    print(f"Sucesso! {ptr} parâmetros carregados perfeitamente.")
    return modelo



def carregar_pesos_ultralytics_v26(caminho_weights, modelo=None, device='cpu'):
    yolo_model = YOLO(caminho_weights)
    yolo_model.to(device)
    
    if modelo is None:
        print("Usando modelo ultralytics nativo")
        return yolo_model
    else:
        modelo = transferir_pesos_ultralytics(yolo_model, modelo)
        return modelo


def transferir_pesos_ultralytics(yolo_ultralytics, modelo_customizado):
    
    ultralytics_state = yolo_ultralytics.model.state_dict()
    custom_state = modelo_customizado.state_dict()
    
    transferred = 0
    for name, param in custom_state.items():
        if name in ultralytics_state:
            if param.shape == ultralytics_state[name].shape:
                custom_state[name] = ultralytics_state[name]
                transferred += 1
    
    modelo_customizado.load_state_dict(custom_state)
    print(f"✓ Transferidos {transferred} parâmetros com sucesso")
    return modelo_customizado


def extrair_backbone_ultralytics(caminho_weights, device='cpu'):
    
    yolo_model = YOLO(caminho_weights)
    yolo_model.to(device)
    
    backbone = yolo_model.model.model[:10]
    return backbone


def salvar_pesos_compativel_ultralytics(modelo, caminho_saida):
    """Salva pesos em formato compatível com ultralytics."""
    print(f"Salvando pesos em: {caminho_saida}")
    
    checkpoint = {
        'model': modelo.state_dict(),
        'epoch': 0,
        'best_fitness': 0.0,
        'date': None,
    }
    
    torch.save(checkpoint, caminho_saida)
    print("✓ Pesos salvos com sucesso")
