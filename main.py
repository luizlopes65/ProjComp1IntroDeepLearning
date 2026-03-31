import cv2
import numpy as np
import os

weights_path = os.path.join("weights", "yolov3.weights")
config_path = os.path.join("cfg","yolov3.cfg" )
data_path = os.path.join("data","coco.names" )

model = cv2.dnn.readNetFromDarknet(config_path,weights_path)

classes_names = []
k = open(data_path,'r')
for i in k.readlines():
    classes_names.append(i.strip())

def processar_imagem(path_img):
    img = cv2.imread(path_img)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img , 1/255 , (320,320) , True , crop = False)

    model.setInput(blob)
    cfg_data = model.getLayerNames()

    #print(cfg_data)
    layer_names = model.getUnconnectedOutLayers()

    outputs = [cfg_data[i-1] for i in layer_names]
    #print(outputs)

    output_data = model.forward(outputs)

    return output_data

print(processar_imagem("images/cat.jpg"))

# essa parte não peguei pra entender ainda
#prediction_box , bounding_box , confidence , class_labels = bounding_box_prediction(output_data)

#inal_prediction(prediction_box , bounding_box , confidence , class_labels , original_with / 320 , original_height / 320 )