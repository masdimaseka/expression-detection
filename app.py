import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model-deteksi-ekspresi-augmented.h5')

emotion_dict = {
    0: "Marah", 1: "Jijik", 2: "Takut", 3: "Senang",
    4: "Netral", 5: "Sedih", 6: "Terkejut"
}

cap = cv2.VideoCapture(0)
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Grayscale (pengolahan awal)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)  # Deteksi wajah

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0:
            continue

        # Resize ke 48x48 dan normalisasi (rescale)
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.expand_dims(np.expand_dims(cropped_img, -1), 0) / 255.0

        # Prediksi ekspresi
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        confidence = np.max(prediction) * 100  # Persentase keyakinan

        # Label dengan confidence
        label_text = f"{emotion_dict[maxindex]} ({confidence:.2f}%)"

        # Gambar bounding box dan label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow('Deteksi Ekspresi Wajah', cv2.resize(frame, (1000, 600)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()