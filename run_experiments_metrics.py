import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
from PIL import Image, ImageDraw, ImageFont
from weights import carregar_pesos_ultralytics_v26, carregar_pesos_yolov3
from model import YOLOv3
from config import YOLOV3_ANCHORS
from utils import (read_classes, preprocess_image, reverter_escala_caixas, 
                   generate_colors, manual_nms)
from inference import decode_yolo

# --- CONFIGURAÇÕES DOS EXPERIMENTOS ---
# Adicione ou remova combinações aqui
GRID_PESQUISA = [
    {'conf': 0.25, 'iou': 0.45},
    {'conf': 0.50, 'iou': 0.45},
    {'conf': 0.25, 'iou': 0.25},
]

def criar_estrutura_pastas():
    Path("exps/comparativos").mkdir(parents=True, exist_ok=True)

def listar_imagens(pasta="images"):
    extensoes = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    imagens = []
    for ext in extensoes:
        imagens.extend(glob(f"{pasta}/{ext}"))
    return sorted(imagens)

def executar_exp_v26(imagens, device, conf, iou):
    """Executa YOLOv26 e retorna métricas médias."""
    modelo = carregar_pesos_ultralytics_v26('yolov8n.pt', device=device)
    tempos = []
    detecoes = []
    
    for img_path in imagens:
        start = time.time()
        results = modelo.predict(source=img_path, conf=conf, iou=iou, verbose=False)
        tempos.append(time.time() - start)
        detecoes.append(len(results[0].boxes))
        
        # Salva apenas uma amostra para não lotar o disco (opcional)
        # results[0].save(f"exps/last_v26.jpg")

    return np.mean(tempos), np.mean(detecoes)

def executar_exp_v3(imagens, device, conf, iou):
    """Executa YOLOv3 e retorna métricas médias."""
    class_names = read_classes("data/coco.names")
    modelo = YOLOv3(num_classes=len(class_names)).to(device)
    # Carregando pesos convertidos
    if Path("yolov3_convertido.pth").exists():
        modelo.load_state_dict(torch.load("yolov3_convertido.pth", map_location=device))
    modelo.eval()

    tempos = []
    detecoes = []

    for img_path in imagens:
        start = time.time()
        image, image_data = preprocess_image(img_path, (416, 416))
        image_data = image_data.to(device)

        with torch.no_grad():
            out1, out2, out3 = modelo(image_data)
            b1, s1 = decode_yolo(out1, YOLOV3_ANCHORS[0], len(class_names), 416)
            b2, s2 = decode_yolo(out2, YOLOV3_ANCHORS[1], len(class_names), 416)
            b3, s3 = decode_yolo(out3, YOLOV3_ANCHORS[2], len(class_names), 416)
            
            all_boxes = torch.cat([b1, b2, b3], dim=0)
            all_scores = torch.cat([s1, s2, s3], dim=0)
            
            box_class_scores, box_classes = torch.max(all_scores, dim=-1)
            mask = box_class_scores >= conf # Usando o CONF do parâmetro
            
            boxes, scores, classes = all_boxes[mask], box_class_scores[mask], box_classes[mask]
            
            if boxes.size(0) > 0:
                keep = manual_nms(boxes, scores, classes, iou) # Usando o IOU do parâmetro
                num_det = len(keep)
            else:
                num_det = 0
        
        tempos.append(time.time() - start)
        detecoes.append(num_det)

    return np.mean(tempos), np.mean(detecoes)

def imprimir_tabela(resultados):
    print("\n" + "="*90)
    print("📊 RESULTADOS COMPARATIVOS: YOLOv3 vs YOLOv26".center(90))
    print("="*90)
    header = f"{'Configuração (C/I)':<20} | {'V3 Time (s)':<12} | {'V26 Time (s)':<12} | {'V3 Det.':<10} | {'V26 Det.':<10}"
    print(header)
    print("-" * 90)
    
    for r in resultados:
        row = (f"Conf:{r['conf']:.2f} IOU:{r['iou']:.2f} | "
               f"{r['v3_t']:<12.4f} | {r['v26_t']:<12.4f} | "
               f"{r['v3_d']:<10.1f} | {r['v26_d']:<10.1f}")
        print(row)
    print("="*90)

def main():
    criar_estrutura_pastas()
    imagens = listar_imagens("images")
    if not imagens: return print("Imagens não encontradas.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Iniciando testes em {len(imagens)} imagens usando {device}...\n")
    
    resultados_finais = []

    for param in GRID_PESQUISA:
        c, i = param['conf'], param['iou']
        print(f"🔎 Testando: Conf={c}, IOU={i}...")
        
        # Execução YOLOv26
        v26_t, v26_d = executar_exp_v26(imagens, device, c, i)
        
        # Execução YOLOv3
        v3_t, v3_d = executar_exp_v3(imagens, device, c, i)
        
        resultados_finais.append({
            'conf': c, 'iou': i,
            'v26_t': v26_t, 'v26_d': v26_d,
            'v3_t': v3_t, 'v3_d': v3_d
        })

    imprimir_tabela(resultados_finais)

if __name__ == "__main__":
    main()
