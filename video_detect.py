import cv2
import numpy as np
from tensorflow.keras.models import load_model

labels_dict = {0: 'mask', 1: 'No mask'}
cap = cv2.VideoCapture(0)
# Window width
cap.set(3, 640)
# Window length
cap.set(4, 480)
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
imgSize = 4
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def predict(model):
    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1, 1)
        imgs = cv2.resize(img, (img.shape[1] // imgSize, img.shape[0] //
                                imgSize))
        face_rec = classifier.detectMultiScale(imgs)
        for i in face_rec:  # Overlay rectangle on face
            (x, y, l, w) = [v * imgSize for v in i]
            face_img = img[y:y + w, x:x + l]
            resized = cv2.resize(face_img, (224, 224))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 224, 224, 3))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(img, (x, y), (x + l, y + w), color_dict[label], 2)
            cv2.rectangle(img, (x, y - 40), (x + l, y), color_dict[label], -1)
            cv2.putText(img, labels_dict[label] + ": " + str(round(result[0][label] * 100, 3)) + "%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('LIVE', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


ResNetModel = load_model("mask_detector_ResNet.model")
predict(ResNetModel)
