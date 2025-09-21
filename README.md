# 🚗 Parking Violation Detection (Deep Learning Models)

This repository contains Jupyter Notebooks for training and evaluating deep learning models to detect and classify **parking violations**.  
The dataset was annotated using [Roboflow](https://app.roboflow.com/), where bounding boxes were created to identify vehicles and determine whether they are parked correctly.

---

## 📂 Project Structure
- `train.ipynb` → End-to-end training pipeline including data preprocessing, model training, and evaluation.  
- `mobileNet.ipynb` → Implementation of **MobileNet** model optimized for lightweight and fast image classification.  
- `modelfastrnn.ipynb` → Implementation of **FastRNN** model designed for sequential and feature-based analysis.  

---

## 📊 Dataset
- **Source**: Custom dataset labeled with Roboflow.  
- **Classes**: Vehicles with violation and non-violation categories.  
- **Format**: YOLO / Pascal VOC / COCO (exported from Roboflow).  
- **Split**: Training / Validation / Testing provided during export.  

---

## ⚙️ Requirements
Install the required dependencies before running the notebooks:

```bash
pip install tensorflow torch torchvision roboflow opencv-python matplotlib seaborn numpy pandas jupyter
