import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import json

# -------------------------
# 1. Load Model Once
# -------------------------
@st.cache(allow_output_mutation=True)  # works with older Streamlit
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes=2  # background + car
    )
    weights_path = "/Users/farahalhanaya/computer-vision-project-mawqif/models_training/faster_rcnn_car.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# 2. Load ROI
# -------------------------
with open("/Users/farahalhanaya/computer-vision-project-mawqif/roi_artifacts/roi_points_normalized.json") as f:
    roi_points_norm = json.load(f)

def denormalize_roi(roi_points_norm, W, H):
    return np.array([[int(x * W), int(y * H)] for (x, y) in roi_points_norm], dtype=np.int32)

def check_roi_overlap(roi_points, car_box, H, W, threshold=0.2):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_points], 255)

    car_mask = np.zeros((H, W), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, car_box)
    cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)

    intersection = cv2.bitwise_and(mask, car_mask)
    inter_area = cv2.countNonZero(intersection)

    car_area = (x2 - x1) * (y2 - y1)
    return inter_area / car_area > threshold

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("🚗 Parking Violation Detector")
st.write("Upload an image to check for cars crossing into forbidden zones (ROI).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    W, H = image.size
    roi_points = denormalize_roi(roi_points_norm, W, H)

    transform = T.ToTensor()
    img_tensor = transform(image)

    # Run detection
    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]["boxes"].numpy()
    scores = outputs[0]["scores"].numpy()
    labels = outputs[0]["labels"].numpy()

    # Draw detections and ROI
    vis = np.array(image.copy())
    cv2.polylines(vis, [roi_points], isClosed=True, color=(0,0,255), thickness=3)

    results = []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5 and label == 1:  # class 1 = car
            x1, y1, x2, y2 = map(int, box)
            violation = check_roi_overlap(roi_points, box, H, W)
            color = (0,0,255) if violation else (0,255,0)
            status = "Violation" if violation else "OK"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{status} {score:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            results.append({
                "box": box.tolist(),
                "score": float(score),
                "violation": violation
            })

    # Show annotated image
    st.image(vis, caption="Detection Results", use_column_width=True)

    # Show raw JSON
    st.subheader("Detection Results (JSON)")
    st.json({"results": results})
