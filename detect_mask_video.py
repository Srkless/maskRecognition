
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import math
import os
import sys
from threading import Timer
import shutil
import time

detections = None 

def detect_and_predict_mask(frame, faceNet, maskNet,threshold):

	global detections 
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence >threshold:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
            
			locs.append((startX, startY, endX, endY))
			preds.append(maskNet.predict(face)[0].tolist())
	return (locs, preds)

MASK_MODEL_PATH=os.getcwd()+"/model/mask_model.h5"
FACE_MODEL_PATH=os.getcwd()+"/face_detector" 
SOUND_PATH=os.getcwd()+"/sounds/alarm.wav" 
THRESHOLD = 0.5

#mixer.init()
#sound = mixer.Sound(SOUND_PATH)

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([FACE_MODEL_PATH, "deploy.prototxt"])
weightsPath = os.path.sep.join([FACE_MODEL_PATH,"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(MASK_MODEL_PATH)

print("[INFO] starting video stream...")
vs = VideoStream(0).start()
time.sleep(2.0)


while True:

	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	original_frame = frame.copy()

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet,THRESHOLD)

	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No mask"
		 
		if (label == "Mask"):
			color = (0, 255, 0)
		else:
			color = (0, 0, 255)

		label = "{}".format(label)

		cv2.putText(original_frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(original_frame, (startX, startY), (endX, endY), color, 2)
		cv2.rectangle(frame, (startX, startY+math.floor((endY-startY)/1.6)), (endX, endY), color, -1)
    

	cv2.addWeighted(frame, 0.5, original_frame, 0.5 , 0,frame)

	frame= cv2.resize(frame,(860,490))
	cv2.imshow("Mask detector by Srkless", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()

    
