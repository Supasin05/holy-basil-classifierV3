import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Holy Basil Classifier (MobileNetV2 + Grad-CAM)",
    page_icon="🌿",
    layout="centered"
)

# โหลดอุปกรณ์ที่ใช้ประมวลผล
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['ใบกะเพราชำรุดหรือมีโรคใบเสีย (Unhealthy / diseased)', 'ใบกะเพราสมบูรณ์ดี (Healthy)']

@st.cache_resource
def load_model():
    # สร้างโครงสร้าง MobileNetV2 ตามการเทรน
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.last_channel, 2))
    
    model_path = "best_basil_mobilenetv2.pt"
    if not os.path.exists(model_path):
        model_path = "best_basil_mobilenetv2_bgaug.pt"
        
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model

model = load_model()
cam = GradCAM(model=model, target_layers=[model.features[-1]])

eval_tf = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🌿 ระบบตรวจจับและจำแนกประเภทใบกะเพรา")
st.markdown("อัปโหลดภาพใบกะเพราเพื่อให้ AI จำแนกประเภท (ดี / เสีย) พร้อมแสดงภาพวิเคราะห์ความโปร่งใสด้วย **Grad-CAM**")

uploaded_file = st.file_uploader("เลือกไฟล์รูปภาพใบกะเพรา...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("📷 **ภาพที่อัปโหลด**")
        st.image(image, use_container_width=True)
        
    # พรีประมวลผลและทำนาย
    img_resized = image.resize((384, 384))
    input_tensor = eval_tf(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item() * 100
        
    # สร้าง Grad-CAM Heatmap
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(np.asarray(img_resized, dtype=np.float32) / 255, grayscale_cam, use_rgb=True)
    
    with col2:
        st.markdown("🔍 **พื้นที่วิเคราะห์ (Grad-CAM)**")
        st.image(visualization, use_container_width=True)
        st.caption("บริเวณสีแดงคือจุดที่โมเดลใช้ในการตัดสินใจ")
        
    st.markdown("---")
    st.markdown("### 📊 สรุปผลการวินิจฉัย")
    
    res_text = f"ผลลัพธ์: {CLASSES[pred_idx]} (ความมั่นใจ {confidence:.2f}%)"
    if pred_idx == 1:
        st.success(res_text)
        st.info("💡 **คำแนะนำ:** ใบกะเพราของคุณมีสภาพสมบูรณ์และพร้อมสำหรับการนำไปบริโภคหรือประกอบอาหารได้อย่างปลอดภัย หมั่นดูแลรดน้ำและให้ปุ๋ยสม่ำเสมอครับ")
    else:
        st.error(res_text)
        st.warning("💡 **คำแนะนำและแนวทางแก้ไข:**\n1. **ตรวจสอบความเสียหาย:** ตรวจดูว่าแผลเกิดจากเชื้อรา เพลี้ย หรือขาดสารอาหาร (สังเกตบริเวณสีแดงเด่นชัดในภาพ Grad-CAM)\n2. **ตัดกิ่ง/ใบที่เสียหายออก:** แนะนำให้ตัดใบที่มีปัญหาทิ้งทันที เพื่อลดการแพร่กระจายของเชื้อโรคไปยังส่วนอื่น\n3. **การรักษา:** หลีกเลี่ยงการรดน้ำที่ใบโดยตรงในช่วงเย็นเพื่อลดความชื้นสะสม และพิจารณาใช้สารชีวภัณฑ์ในการกำจัดโรคพืช")
