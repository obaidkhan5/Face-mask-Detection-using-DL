import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("face_mask_model.keras")

# Face detector
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128, 128))
        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)

        confidence = np.max(pred)

        label = np.argmax(pred)

        if label == 0:

            text = f"MASK {confidence*100:.1f}%"
            color = (0, 255, 0)

        else:

            text = f"NO MASK {confidence*100:.1f}%"
            color = (0, 0, 255)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            3
        )

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Mask Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()