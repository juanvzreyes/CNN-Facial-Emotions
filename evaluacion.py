import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Ruta al modelo guardado
modelo_path = r"EI_Emociones/modelo/modelo.h5"

# Cargar modelo
CNN = load_model(modelo_path)

# Etiquetas de emociones
emociones = ['Enojado', 'Disgustado', 'Asustado', 'Feliz', 'Neutral', 'Triste', 'Sorprendido']

# Configurar cámara
cap = cv2.VideoCapture(0) #se usa 1 porque es camara remota desde celular

# Cargar clasificador Haar para detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables para suavizar predicciones en tiempo real
predicciones_buffer = []
buffer_size = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises para detección
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Recortar y preprocesar ROI (Región de Interés)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))  # Ajustar al tamaño esperado por el modelo
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predecir emoción
        pred = CNN.predict(roi)[0]
        predicciones_buffer.append(pred)
        if len(predicciones_buffer) > buffer_size:
            predicciones_buffer.pop(0)

        # Promediar predicciones para suavizar
        pred_avg = np.mean(predicciones_buffer, axis=0)
        emocion = emociones[np.argmax(pred_avg)]
        confianza = np.max(pred_avg) * 100

        # Dibujar rectángulo y texto
        color = (0, 255, 0)  # Verde para bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame,
                    f"{emocion} ({confianza:.1f}%)",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2)

    cv2.imshow('Detección de Emociones', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
