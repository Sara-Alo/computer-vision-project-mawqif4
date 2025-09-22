import cv2
import json
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import argparse
import os

#.\venv\Scripts\activate
#python fixed_roi.py --video "C:/Users/Lenovo/OneDrive/Desktop/CV_project/computer-vision-project-mawqif/models_training/video.mp4" --weights "C:/Users/Lenovo/OneDrive/Desktop/CV_project/computer-vision-project-mawqif/faster_rcnn_car.pth" --out "output1.mp4"


#Command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--out', type=str, default="output.mp4")
args = parser.parse_args()

video_path = args.video
weights_path = args.weights
output_path = args.out

if not os.path.exists(video_path):
    print(f" ERROR: Video file not found: {video_path}")
    exit()

# Select multiple ROIs
roi_list = []  
current_roi = []

def draw_points(event, x, y, flags, param):
    global current_roi, frame_copy
    if event == cv2.EVENT_LBUTTONDOWN:
        current_roi.append((x, y))
        cv2.circle(frame_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("ROI Selector", frame_copy)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print(" ERROR: Cannot read the first frame of the video!")
    exit()

frame_copy = frame.copy()
cv2.namedWindow("ROI Selector", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selector", 800, 600)
cv2.imshow("ROI Selector", frame_copy)
cv2.setMouseCallback("ROI Selector", draw_points)

print(" Left-click = add ROI points")
print("Press ENTER = save ROI")
print("Press N = start new ROI")
print("Press ESC = finish selection")

while True:
    key = cv2.waitKey(0)
    if key == 13:  # ENTER
        if len(current_roi) >= 3:
            roi_list.append(current_roi)
            print(f" ROI saved: {current_roi}")
        current_roi = []
    elif key == ord('n'):
        if len(current_roi) >= 3:
            roi_list.append(current_roi)
            print(f" ROI saved: {current_roi}")
        current_roi = []
        frame_copy = frame.copy()
        for roi in roi_list:
            cv2.polylines(frame_copy, [np.array(roi, np.int32)], True, (0, 0, 255), 2)
        cv2.imshow("ROI Selector", frame_copy)
    elif key == 27:  # ESC
        if len(current_roi) >= 3:
            roi_list.append(current_roi)
        break

cv2.destroyAllWindows()

if roi_list:
    with open("roi_points.json", "w") as f:
        json.dump(roi_list, f)
    print(f" {len(roi_list)} ROIs saved")
else:
    print("No ROI saved, exiting...")
    exit()


# Load  model
def load_model(weights_path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes=2
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

if not os.path.exists(weights_path):
    print(f" ERROR: Weights file not found: {weights_path}")
    exit()

model = load_model(weights_path)
print(" Model loaded successfully")

#  ROI violation check
def check_roi_overlap_and_violation_strict(roi_points, car_box, H, W):
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [roi_points], 255)

    x1, y1, x2, y2 = map(int, car_box)
    car_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(car_mask, (x1, y1), (x2, y2), 255, -1)
    
    car_area = (x2 - x1) * (y2 - y1)
    if car_area == 0:
        return False

    intersection_mask = cv2.bitwise_and(roi_mask, car_mask)
    intersection_area = cv2.countNonZero(intersection_mask)
    inside_ratio = intersection_area / car_area

    return inside_ratio == 1.0

# Process video
with open("roi_points.json") as f:
    roi_list = json.load(f)
    roi_list = [np.array(roi, dtype=np.int32) for roi in roi_list]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f" ERROR: Cannot open video: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                      (frame_width, frame_height))

transform = T.ToTensor()
display_width = 800
display_height = int(frame_height * (display_width / frame_width))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img)

    with torch.no_grad():
        outputs = model([img_tensor])

    boxes = outputs[0]["boxes"].numpy()
    scores = outputs[0]["scores"].numpy()
    labels = outputs[0]["labels"].numpy()

    for roi in roi_list:
        cv2.polylines(frame, [roi], True, (0, 0, 255), 3)

    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5 and label == 1:
            x1, y1, x2, y2 = map(int, box)
            violation = any(check_roi_overlap_and_violation_strict(roi, box, frame_height, frame_width) for roi in roi_list)
            color = (0, 0, 255) if violation else (0, 255, 0)
            status = "Violation" if violation else "OK"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{status} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    frame_display = cv2.resize(frame, (display_width, display_height))
    cv2.imshow("Video Detection", frame_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Finished! Video saved as {output_path}")
