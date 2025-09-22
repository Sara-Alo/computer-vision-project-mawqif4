import streamlit as st
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import json
from twilio.rest import Client
import os


# -------------------------
# 1. Load Model Once
# -------------------------
@st.cache_resource
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes=2  # background + car
    )
    weights_path = "faster_rcnn_car.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------
# 2. Load ROI
# -------------------------
with open("roi_artifacts/roi_points_normalized.json") as f:
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
# Twilio WhatsApp Setup 
# -------------------------
twilio_account_sid = "ACda9d1a900ee4cbba2f57326e933dd4e0"  
twilio_auth_token  = "0fd71a11eb8477357f3a41d1668ca612"    
source_whatsapp_number = "whatsapp:+14155238886"
destination_whatsapp_number = "whatsapp:+966533186523"  

twilio_client = Client(twilio_account_sid, twilio_auth_token)

# -------------------------
# 3. Streamlit UI
# -------------------------
st.title("🚗 Mawqif (مواقف)")

tab1, tab2, tab3, tab4 = st.tabs(["About", 'Dataset',"Prediction", "Team"])

# -------------------------
# TAB 1: About
# -------------------------
with tab1:
    st.markdown("## Parking Violation Detector Overview")
    st.markdown("### Problem")
    st.write(
        "With the increasing population of Riyadh and the city's heavy traffic, parking has become a major problem. "
        "Many drivers do not park their cars correctly, and it is impractical for security personnel to photograph each car and issue violations manually."
    )
    st.markdown("### Our Solution")
    st.write(
        "We developed a smart system that detects cars parked incorrectly in parking lots and automatically records violations.\n"
        "The system sends a notification to the car owner with the issued violation and the corresponding fine.\n"
        "The goal is to improve compliance with parking regulations and organize parking in crowded cities like Riyadh."
    )
    st.markdown("### Workflow / Model Pipeline")
    st.write(
        "1. **Data Collection**: Collected images of parking lots from Roboflow.\n"
        "2. **Data Annotation**: Labeled car positions and parking lines in the images.\n"
        "3. **Model Training & Testing**:\n"
        "   - Tested multiple models: MobileNet, Faster R-CNN, YOLOv8, YOLOv11.\n"
        "   - Chose Faster R-CNN because it provided the highest accuracy in car detection.\n"
        "4. **Region of Interest (ROI)**: Defined the parking area for each spot in the image.\n"
        "5. **Violation Detection**: Used the `check_roi_overlap` function to check whether a car is within the allowed area.\n"
        "6. **Notifications**: If a violation occurs, the system automatically sends a message to the car owner."
    )

# -------------------------
# TAB 2: Dataset
# -------------------------
with tab2:
    st.markdown("## Dataset")
    st.markdown("### Dataset Description")
    st.write(
        """
In this project, we used the **Parking Computer Vision Dataset** from Roboflow. The dataset contains a total of **689 images** of parking areas, each annotated with **bounding boxes around vehicles** to precisely indicate their locations. The annotation process was done using Roboflow’s labeling tool, which allowed us to clearly define where each car is located in the scene.

To ensure effective model training and evaluation, the dataset was divided into three parts:

- **Training set (70%)**  
- **Validation set (20%)**  
- **Test set (10%)**  

This dataset contains **images with different lighting conditions, vehicle types, and parking angles**, which helped the model generalize better.
"""
    )

    # Load COCO Annotations
    base_path = "Dataset\train"  
    annotations_file = os.path.join(base_path, "_annotations.coco.json")

    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"File not found: {annotations_file}")

    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    img_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

    dataset_dict = {}
    for ann in coco['annotations']:
        img_filename = img_id_to_filename[ann['image_id']]
        if img_filename not in dataset_dict:
            dataset_dict[img_filename] = []
        dataset_dict[img_filename].append({
            'bbox': ann['bbox'], 
            'category_id': ann['category_id']
        })

    # Display Sample Images with Boxes
    st.markdown("### Sample Images with Bounding Boxes")
    sample_images = list(dataset_dict.items())[:5]

    for img_name, boxes in sample_images:
        img_path = os.path.join(base_path, img_name)
        if not os.path.exists(img_path):
            st.warning(f"Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            st.warning(f"Unable to read image: {img_path}")
            continue

        for box in boxes:
            x, y, w, h = box['bbox']
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption=f"{img_name} - Boxes: {len(boxes)}", use_column_width=True)

# -------------------------
# TAB 3: Prediction
# -------------------------
with tab3:
    st.write("Upload an image to check for cars crossing into forbidden zones (ROI).")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        W, H = image.size
        roi_points = denormalize_roi(roi_points_norm, W, H)

        img_tensor = T.ToTensor()(image)

        with torch.no_grad():
            outputs = model([img_tensor])

        boxes = outputs[0]["boxes"].numpy()
        scores = outputs[0]["scores"].numpy()
        labels = outputs[0]["labels"].numpy()

        vis = np.array(image.copy())
        cv2.polylines(vis, [roi_points], isClosed=True, color=(0,0,255), thickness=3)

        recipient_mobile_number = "+966533186523"
        results = []

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
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

                # إرسال WhatsApp إذا في مخالفة
                if violation:
                    try:
                        message = twilio_client.messages.create(
                            body=(
                                "⚠️ إشعار مخالفة موقف\n"
                                "نوع المخالفة: الوقوف غير الصحيح خارج المكان المخصص.\n"
                                f"رقم المخالف: {recipient_mobile_number}\n"
                                "قيمة الغرامة: 800 ريال"
                            ),
                            from_=source_whatsapp_number,
                            to=destination_whatsapp_number
                        )
                        st.success(f"📩 WhatsApp message sent to {recipient_mobile_number} (SID: {message.sid})")
                    except Exception as e:
                        st.error(f" Error sending WhatsApp message: {e}")

        st.image(vis, caption="Detection Results", use_container_width=True)

    image_path = "IMG2.png"
    image = Image.open(image_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)


    st.markdown("### 🚨 Example of a Parking Violation 🚨")

    output_video_path = "output1.mp4"

    if os.path.exists(output_video_path):
        st.video(output_video_path)
    else:
        st.warning(" Video not found at the specified path.")

# TAB 4: Team
# -------------------------
with tab4:
    st.markdown('<p class="section-header" style="font-size:28px; font-weight:bold;">Team Members</p>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)

        member_box_style = """
        border:1px solid #2196F3; 
        border-radius:12px; 
        padding:20px; 
        margin:10px 0; 
        background:#E3F2FD;
        height:180px; 
        display:flex; 
        flex-direction:column; 
        justify-content:center;
        text-align:center;
        """

        # Member 1
        with col1:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#0D47A1">Sarah Alowjan</h4>
            </div>
            """, unsafe_allow_html=True)

        # Member 2
        with col2:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#0D47A1">Farah Alhanaya</h4>
            </div>
            """, unsafe_allow_html=True)

        # Member 3
        with col3:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#0D47A1"> Aljwharah  Almousa</h4>
            </div>
            """, unsafe_allow_html=True)

        # Member 4
        with col4:
            st.markdown(f"""
            <div style="{member_box_style}">
              <h4 style="margin:0 0 8px 0; color:#0D47A1"> Rawabi Almutairi</h4>
            </div>
            """, unsafe_allow_html=True)

    
    # Contributions Section
    st.markdown('<p class="section-header" style="font-size:28px; font-weight:bold;">Team Contributions</p>', unsafe_allow_html=True)

    st.markdown("""
  
    ### Farah Alhanaya
    - Dataset research and annotation for car detection.
    - Developed the **MobileNet** and **R-CNN**model for car detection.
    - Assisted with **Region of Interest (ROI)** implementation to define parking slots along with Sarah and Aljwharah.
    - Contributed to designing the **Streamlit interface** with Sarah and Aljwharah.

    ### Aljwharah  Almousa
    - Dataset research and annotation for car detection.
    - Developed the **Faster R-CNN** model for car detection.
    - Assisted with **Region of Interest (ROI)** implementation with Farahand sarah to define parking slots.
    - Contributed to designing the **Streamlit interface** with Farah and Sarah.

    ### Rawabi Almutairi
    - Dataset research and annotation for car detection.
    - Developed **YOLOv8 and YOLOv11** models for car detection.
    - Prepared the **project README** file.

    ### Sarah Alowjan
    - Dataset research and annotation for car detection.
    - Developed the **RF-DETR** model using Roboflow.
    - Assisted with **Region of Interest (ROI)** implementation with Farah and Aljwharah to define parking slots.
    - Integrated **violation notifications with WhatsApp** using Twilio to send alerts automatically.
    - Contributed to designing the **Streamlit interface** with Farah and Aljwharah.
    """)
