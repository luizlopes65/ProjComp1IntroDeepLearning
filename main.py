import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from helpers import processar_imagem, bounding_box_prediction, final_prediction

# paths
weights_path = os.path.join("weights", "yolov3.weights")
config_path = os.path.join("cfg", "yolov3.cfg")
data_path = os.path.join("data", "coco.names")

model = cv2.dnn.readNetFromDarknet(config_path, weights_path)

classes_names = []
with open(data_path, 'r') as f:
    for line in f.readlines():
        classes_names.append(line.strip())

# processar imagem
image_path = 'images'
output_data, image = processar_imagem(image_path, model)

original_width, original_height = image.shape[1], image.shape[0]

#gera a prefição
prediction_box, bounding_box, confidence, class_labels = bounding_box_prediction(output_data)

# gera a imagem desenhada
result_image = final_prediction(
    image, 
    prediction_box, 
    bounding_box, 
    confidence, 
    class_labels, 
    classes_names, 
    original_width / 320, 
    original_height / 320
)

output_path = 'images/output.jpg'
cv2.imwrite(output_path, result_image)

result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 8))
plt.imshow(result_image_rgb)
plt.title('YOLOv3 Detection')
plt.axis('off')
plt.tight_layout()
plt.show()

