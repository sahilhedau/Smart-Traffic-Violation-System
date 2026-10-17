# Google Colab GPU Training Guide (Free NVIDIA T4 GPU)
**Smart Traffic Violation Detection System (TrafficGuard AI)**

If training on your local CPU is slow or your laptop doesn't have an NVIDIA CUDA GPU, you can train your models in **10–15 minutes** for free on Google Colab.

---

### Step 1: Open Google Colab
1. Go to [https://colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. In the menu, go to **Runtime** > **Change runtime type**, select **T4 GPU**, and click **Save**.

---

### Step 2: Install Ultralytics
In the first Colab cell, run:
```python
!pip install -q ultralytics
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
```

---

### Step 3: Train Helmet Detection Model
#### Option A: Using Roboflow directly in Colab
```python
!pip install -q roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("marcvador").project("bike-helmets-detection")
version = project.version(1)
dataset = version.download("yolov8")

# Train YOLOv8n on GPU
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=0,
    name="helmet_colab_run"
)
```

#### Option B: Upload your zip file
1. Zip `Bike-Helmets-Detection.v1i.yolov5pytorch` into `helmet_dataset.zip`.
2. Upload it to Colab by clicking the file folder icon on the left pane and dragging the zip.
3. Run:
```python
!unzip -q helmet_dataset.zip -d /content/helmet_dataset

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="/content/helmet_dataset/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    device=0,
    name="helmet_colab_run"
)
```

---

### Step 4: Download the Trained Weights
In Colab, run:
```python
from google.colab import files
files.download('/content/runs/detect/helmet_colab_run/weights/best.pt')
```

Rename the downloaded file to:
`helmet_yolov8n.pt`
and place it in your project's `models/` directory:
`d:\Smart-Traffic-Violation-System\models\helmet_yolov8n.pt`

TrafficGuard AI will automatically detect and load it on next startup!
