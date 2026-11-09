import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Holy Basil Health Classifier",
    page_icon="🌿",
    layout="centered"
)

# คลาสและคอนฟิกูเรชันพื้นฐาน
CLASSES = ['bad', 'good']
IMG_SIZE = 384
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# โหลดโมเดล
@st.cache_resource
def load_model():
    # ลิงก์พาทของไฟล์ checkpoint ที่บันทึกไว้
    ckpt_path = '/content/best_basil_mobilenetv2_bgaug.pt'
    if not os.path.exists(ckpt_path):
        # ลองหาใน Google Drive สำรอง
        ckpt_path = '/content/drive/MyDrive/basil/best_basil_mobilenetv2_bgaug.pt'
        
    if not os.path.exists(ckpt_path):
        st.error("❌ ไม่พบไฟล์โมเดลกรุณาตรวจสอบว่ามีไฟล์โมเดลที่ฝึกเสร็จแล้วหรือยัง")
        return None, None
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torchvision.models as models
    model = models.mobilenet_v2()
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, 2))
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ฟังก์ชันเตรียมภาพ
def preprocess_image(pil_img):
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return eval_tf(pil_img).unsqueeze(0).to(device)

# ส่วนของหน้าเว็บหลัก
st.title("🌿 ระบบตรวจสุขภาพใบกะเพรา")
st.write("อัปโหลดรูปภาพใบกะเพราเพื่อจำแนกว่าเป็น **ใบดี (Good)** หรือ **ใบเสีย (Bad)** พร้อมแสดงผลวิเคราะห์ด้วย Grad-CAM")

uploaded_file = st.file_uploader("เลือกรูปภาพใบกะเพรา (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # แสดงภาพต้นฉบับ
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ ภาพต้นฉบับ")
        st.image(image, use_container_width=True)

    # ทำนายผล
    with st.spinner("กำลังวิเคราะห์ภาพ..."):
        input_tensor = preprocess_image(image)
        
        # ทำนายค่าความน่าจะเป็น
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_class = CLASSES[pred_idx]
            confidence = probs[pred_idx]
            
        # คำนวณ Grad-CAM
        # โมเดล MobileNetV2 layer สุดท้ายอยู่ใน model.features[-1]
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        
        # สร้างภาพทับซ้อน (Overlay)
        resized_img = image.resize((IMG_SIZE, IMG_SIZE))
        img_np = np.asarray(resized_img, dtype=np.float32) / 255.0
        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        
    with col2:
        st.subheader("🔥 Grad-CAM")
        st.image(cam_image, use_container_width=True, caption="บริเวณที่โมเดลใช้พิจารณา")

    # แสดงผลลัพธ์
    st.subheader("📊 ผลการวิเคราะห์")
    if pred_class == 'good':
        st.success(f"**ใบดี (Healthy Holy Basil Leaf)** ด้วยความมั่นใจ {confidence:.2%}")
    else:
        st.error(f"**ใบเสีย (Unhealthy / Diseased Holy Basil Leaf)** ด้วยความมั่นใจ {confidence:.2%}")
        
    # ตารางความน่าจะเป็นของแต่ละคลาส
    st.write("**รายละเอียดคะแนน:**")
    for idx, name in enumerate(CLASSES):
        st.progress(float(probs[idx]), text=f"{name}: {probs[idx]:.2%}")
