import cv2
import mediapipe as mp

# Initialize FaceMesh with support for up to 3 faces
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=3)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    results = face_mesh.process(rgb_frame)

    # If faces are found, draw landmarks
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            for point in landmarks.landmark:
                x = int(point.x * img.shape[1])
                y = int(point.y * img.shape[0])
                cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

    # Display the output
    cv2.imshow("my_video", img)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
