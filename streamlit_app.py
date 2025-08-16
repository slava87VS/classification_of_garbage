import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# === Настройки ===
IMG_SIZE = (192, 256)

# 1. Трансформации
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# DATA_PATH = '../../class_trash/dataset_marker'
# dataset = ImageFolder(root=DATA_PATH, transform=transform)
CLASS_NAMES = ['alu_41', 'c_ldpe_90', 'c_pap_84', 'c_pet', 'c_pp_folg', 'glass_dark', 'glass_transparent', 'green_glass__pap_22', 'hdpe_2__pet_1', 'hdpe_2__pp_5soft__pet_1', 'hdpe_2_soft', 'hdpe_2_solid', 'hdpe_2_solid__pap_20', 'jb_cover', 'jb_cover__glass_dark', 'ldpe_4_color', 'not_defined', 'other_7', 'pap_20', 'pap_21', 'pap_22', 'pet_1', 'pet_1__pap_22', 'pet_1__pap_22__hdpe_2_solid', 'pp_5_folg', 'pp_5_soft', 'pp_5_soft__pet_1', 'pp_5_solid', 'pp_5_solid__c_ldpe_90', 'ps_6_soft', 'ps_6_solid', 'ps_6_solid__pap_22', 'pvc_3', 'pvc_3__ps_6']


# Словарь с описаниями классов
CLASS_DESCRIPTIONS = {
    "alu_41": "Код переработки алюминия 41",
    "c_ldpe_90": "Переработка LDPE (низкотемпературный полиэтилен) 90",
    "c_pap_84": "Переработка бумаги 84",
    "c_pet": "Переработка PET-пластика",
    "c_pp_folg": "Переработка фольги из полипропилена (PP)",
    "glass_dark": "Переработка тёмного стекла",
    "glass_transparent": "Переработка прозрачного стекла",
    "glass_green__pap_22": "Переработка зелёного стекла с бумагой 22",
    "hdpe_2__pet_1": "Переработка HDPE и PET пластика",
    "hdpe_2__pp_5soft__pet_1": "Переработка HDPE, мягкого PP и PET",
    "hdpe_2_soft": "Переработка мягкого HDPE пластика",
    "hdpe_2_solid": "Переработка твёрдого HDPE пластика",
    "hdpe_2_solid__pap_20": "Переработка твёрдого HDPE с бумагой 20",
    "jb_cover": "Переработка крышек JB",
    "jb_cover__glass_dark": "Переработка крышек JB и тёмного стекла",
    "ldpe_4_color": "Переработка цветного LDPE 4",
    "not_defined": "Класс не определён",
    "other_7": "Прочие материалы 7",
    "pap_20": "Переработка бумаги 20",
    "pap_21": "Переработка бумаги 21",
    "pap_22": "Переработка бумаги 22",
    "pet_1": "Переработка PET пластика 1",
    "pet_1__pap_22": "Переработка PET с бумагой 22",
    "pet_1__pap_22__hdpe_2_solid": "Переработка PET, бумаги 22 и твёрдого HDPE",
    "pp_5_folg": "Переработка фольги из PP 5",
    "pp_5_soft": "Переработка мягкого PP 5",
    "pp_5_soft__pet_1": "Переработка мягкого PP 5 и PET",
    "pp_5_solid": "Переработка твёрдого PP 5",
    "pp_5_solid__c_ldpe_90": "Переработка твёрдого PP 5 и LDPE 90",
    "ps_6_soft": "Переработка мягкого PS 6",
    "ps_6_solid": "Переработка твёрдого PS 6",
    "ps_6_solid__pap_22": "Переработка твёрдого PS 6 и бумаги 22",
    "pvc_3": "Переработка PVC 3",
    "pvc_3__ps_6": "Переработка PVC 3 и PS 6",
}

@st.cache_resource
def load_model():
    model = torch.jit.load("trashnet_mobile.pt", map_location='cpu')
    model.eval()
    return model

model = load_model()

# === Интерфейс ===
st.title("🗑️ Классификация мусора по фото")
uploaded_file = st.file_uploader("Загрузите изображение мусора", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)


    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_index = output.argmax(1).item()
        pred_class = CLASS_NAMES[pred_index]
        description = CLASS_DESCRIPTIONS.get(pred_class, pred_class)  # если описание нет — выводим просто метку
        st.success(f"🧠 Предсказанный класс: **{description}**")
