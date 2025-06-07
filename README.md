# 👁️ ComputerVision

ComputerVision is a simple collection of real-time computer vision projects using Python. These programs use your webcam to detect hand distance, recognize facial emotions, and visualize facial landmarks.

## 🔧 Projects Included

- **Hand Distance Alert**  
  Measures the distance between your hand and the webcam. Plays a sound alert if the hand is too close.

- **Face Emotion Detection**  
  Detects and displays the dominant emotion on your face using DeepFace.

- **Face Mesh Visualization**  
  Displays a real-time mesh of 468 facial landmarks using MediaPipe.

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
````

## 🚀 How to Run

```bash
python hand_distance_alert.py
python face_emotion_detection.py
python face_mesh_visualization.py
```

Make sure your webcam is enabled. Press `q` to quit.

## 📁 Files in This Repo

```
ComputerVision/
├── hand_distance_alert.py
├── face_emotion_detection.py
├── face_mesh_visualization.py
├── .gitignore
├── README.md
```

## ✅ Notes

* Works best in well-lit environments.
* Alert sound uses `pygame` (cross-platform) or `winsound` (Windows only).
* Ideal for learning and demonstrating basic computer vision tasks.
