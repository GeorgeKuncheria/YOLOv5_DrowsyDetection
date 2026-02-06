# YOLOv5 Drowsy Detection

This project implements a real-time Drowsy Detection system using the **YOLOv5** architecture. The model is capable of distinguishing between "Awake" and "Drowsy" states by processing live video feeds.

## 🚀 Features
* **Custom Dataset:** Images captured manually to ensure real-world environment accuracy.
* **Manual Annotation:** Data labeled using `labelImg` for high-quality bounding box coordination.
* **Transfer Learning:** Utilizes the YOLOv5 small (`yolov5s`) architecture for a balance between speed and accuracy.
* **Real-time Inference:** Integrated with OpenCV for live webcam monitoring.

---

## 🛠️ Installation & Setup

### 1. Clone Repositories
Clone the main YOLOv5 repository and the labeling tool required for data preparation:

```bash
# Clone YOLOv5
git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

# Clone labelImg for image annotation
git clone [https://github.com/HumanSignal/labelImg.git](https://github.com/HumanSignal/labelImg.git)
```



### 2. Install Dependencies
Ensure you have Python installed, then install the necessary libraries:

```bash
pip install torch torchvision torchaudio
pip install -r yolov5/requirements.txt
pip install pyqt5 lxml # Required for labelImg
```

 ## 📸 Data Pipeline
Data Collection
  
  Images were captured using cv2.VideoCapture to create a personalized dataset.
    1. Classes: Awake, Drowsy
    2. Storage: Images are organized into a directory structure compatible with YOLOv5 training (images and labels).
  
Annotation
  Manual labeling was performed using labelImg:
    1. Open labelImg.
    2. Set the "Save Format" to YOLO.
    3. Draw bounding boxes around the face/eyes and assign the appropriate class.



## 🧠 Training
The model was trained using the following configuration:
 1. Base Model: yolov5s.pt

 2. Image Size: 320

 3. Epochs: 500 (adjust based on loss convergence)

```bash
python yolov5/train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt
```


## 💻 Usage
Loading the Custom Model
  After training, you can load your custom weights (last.pt or best.pt) using Torch Hub:

    ```
        import torch

        # Load local custom model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/last.pt', force_reload=True)
    
    ```

 ### Real-time Detection
       
       Run the following script to start the webcam feed and detect drowsiness:

       ```
       import cv2
       import numpy as np
       
       cap = cv2.VideoCapture(0)
       while cap.isOpened():
           ret, frame = cap.read()
           
           # Make detections 
           results = model(frame)
           
           # Render results on frame
           cv2.imshow('Drowsy Detection', np.squeeze(results.render()))
           
           if cv2.waitKey(10) & 0xFF == ord('q'):
               break
       
       cap.release()
       cv2.destroyAllWindows()

       ```

   ## 📁 Project Structure
          
          Drowsy_Detection_YOLOv5.ipynb: Full development pipeline.

          yolov5/: Detection framework.

          data/: Contains captured images and YOLO format labels.
