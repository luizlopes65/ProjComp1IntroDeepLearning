# ==========================================
# CONFIGURAÇÕES E CONSTANTES DO YOLOV3
# ==========================================

# O YOLOv3 divide a detecção em 3 escalas diferentes (Pequenos, Médios, Grandes)
YOLOV3_ANCHORS = [
    [(116, 90), (156, 198), (373, 326)],  # Escala 1 (13x13) - Objetos Maiores
    [(30, 61), (62, 45), (59, 119)],      # Escala 2 (26x26) - Objetos Médios
    [(10, 13), (16, 30), (33, 23)]        # Escala 3 (52x52) - Objetos Menores
]

 
