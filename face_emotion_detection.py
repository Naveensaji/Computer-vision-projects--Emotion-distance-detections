import cv2
from deepface import DeepFace   # type: ignore

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        dominant_emotion = results[0]['dominant_emotion']

        cv2.putText(img, f'Emotion: {dominant_emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print("Analysis error:", e)

    cv2.imshow("Emotion Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()