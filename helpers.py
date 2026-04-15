import cv2
import numpy as np

# Constantes
THRESHOLD = 0.5
IMAGE_SIZE = 320
FONT = cv2.FONT_HERSHEY_COMPLEX

def bounding_box_prediction(output_data, threshold=THRESHOLD, image_size=IMAGE_SIZE):
    bounding_box = []
    class_labels = []
    confidence_score = []
    for i in output_data:
        for j in i:
            high_label = j[5:]
            classes_ids = np.argmax(high_label)
            confidence = high_label[classes_ids]
            
            if confidence > threshold:
                w , h = int(j[2] * image_size) , int(j[3] * image_size)
                x , y = int(j[0] * image_size - w/2) , int(j[1] * image_size - h/2)
                bounding_box.append([x,y,w,h])
                class_labels.append(classes_ids)
                confidence_score.append(confidence)

    prediction_boxes = cv2.dnn.NMSBoxes(bounding_box , confidence_score , threshold , 0.6)
    return prediction_boxes , bounding_box , confidence_score, class_labels


def processar_imagem(path_img, model, image_size=IMAGE_SIZE):
    img = cv2.imread(path_img)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img , 1/255 , (image_size, image_size) , True , crop = False)

    model.setInput(blob)
    cfg_data = model.getLayerNames()

    #print(cfg_data)
    layer_names = model.getUnconnectedOutLayers()

    outputs = [cfg_data[i-1] for i in layer_names]
    #print(outputs)

    output_data = model.forward(outputs)

    return output_data, img


def final_prediction(image, prediction_box, bounding_box, confidence, class_labels, classes_names, width_ratio, height_ratio, font=FONT):
    for j in prediction_box.flatten():
        x, y, w, h = bounding_box[j]
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        label = str(classes_names[class_labels[j]])
        conf_ = str(round(confidence[j], 2))
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, label + ' ' + conf_, (x, y-2), font, 0.5, (0, 255, 0), 1)
    
    return image