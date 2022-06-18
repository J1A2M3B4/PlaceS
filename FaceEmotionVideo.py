## Detección de emociones en tiempo real ##
##            Vision Artifical 		     ##
##             Equipo PlaceS			 ##

# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time

#Variables para incrementar valores de emociones
i = 0
suma = 0
var1 = 0
var2 = 0
var3 = 0
var4 = 0
var5 = 0
var6 = 0
var7 = 0

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Tipos de emociones del detector
classes = ['Enojado','Disgustado','Temeroso','Feliz','Neutral','Aburrido','Sorprendido']

# Cargamos el  modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Se crea la captura de video
cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)


# Toma la imagen, los modelos de detección de rostros y mascarillas 
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame,faceNet,emotionModel):
	# Construye un blob de la imagen
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

	# Realiza las detecciones de rostros a partir de la imagen
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Listas para guardar rostros, ubicaciones y predicciones
	faces = []
	locs = []
	preds = []
	
	# Recorre cada una de las detecciones
	for i in range(0, detections.shape[2]):
		
		# Fija un umbral para determinar que la detección es confiable
		# Tomando la probabilidad asociada en la deteccion

		if detections[0, 0, i, 2] > 0.4:
			# Toma el bounding box de la detección escalado
			# de acuerdo a las dimensiones de la imagen
			box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
			(Xi, Yi, Xf, Yf) = box.astype("int")

			# Valida las dimensiones del bounding box
			if Xi < 0: Xi = 0
			if Yi < 0: Yi = 0
			
			# Se extrae el rostro y se convierte BGR a GRAY
			# Finalmente se escala a 224x244
			face = frame[Yi:Yf, Xi:Xf]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
			face = cv2.resize(face, (48, 48))
			face2 = img_to_array(face)
			face2 = np.expand_dims(face2,axis=0)

			# Se agrega los rostros y las localizaciones a las listas
			faces.append(face2)
			locs.append((Xi, Yi, Xf, Yf))

			pred = emotionModel.predict(face2)
			preds.append(pred[0])

	return (locs,preds)

while True:
	# Se toma un frame de la cámara y se redimensiona
	im = cv2.imread("places.jpg")
	ret, frame = cam.read()
	frame = imutils.resize(frame, width=1080,height=720)
	(locs, preds) = predict_emotion(frame,faceNet,emotionModel)

	# Para cada hallazgo se dibuja en la imagen el bounding box y la clase
	for (box, pred) in zip(locs, preds):


		(Xi, Yi, Xf, Yf) = box
		(angry,disgust,fear,happy,neutral,sad,surprise) = pred

		#Se calcula la emocion y el porcentaje por separado
		label = ''
		# Se agrega la probabilidad en el label de la imagen
		label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(angry,disgust,fear,happy,neutral,sad,surprise) * 100)
		emo = "{}".format(classes[np.argmax(pred)])
		por = "{:.0f}".format(max(angry, disgust, fear, happy, neutral, sad, surprise) * 100)
		i = i + 1

		#Se obtiene una suma del porcentaje de la emocion obtenida

		if emo == 'Enojado':
			var1 = var1 + int(por)

		if emo == 'Disgustado':
			var2 = var2 + int(por)

		if emo == 'Temeroso':
			var3 = var3 + int(por)

		if emo == 'Feliz':
			var4 = var4 + int(por)

		if emo == 'Neutral':
			var5 = var5 + int(por)

		if emo == 'Triste':
			var6 = var6 + int(por)

		if emo == 'Sorprendido':
			var7 = var7 + int(por)

		#Se imprimen los porcentajes obtenidos
		print('')
		print('Promedio de las emociones detectadas')
		print('Enojado    : ',var1 / i,'%')
		print('Disgustado : ',var2 / i,'%')
		print('Temeroso   : ',var3 / i,'%')
		print('Feliz      : ',var4 / i,'%')
		print('Neutral    : ',var5 / i,'%')
		print('Triste     : ',var6 / i,'%')
		print('Sorprendido: ',var7 / i,'%')
		print('')

		#Se crea el cuadro y el texto que siguen las caras
		cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (68, 252, 124), -1)
		cv2.putText(frame, label, (Xi+5, Yi-15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,127,11), 2)
		cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (68, 252, 124), 3)

	#Se crea una ventana mostrando la webcam
	cv2.imshow('Frame',frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		 break

cv2.destroyAllWindows()
cam.release()