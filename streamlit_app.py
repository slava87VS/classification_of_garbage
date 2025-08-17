import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import json

# === Настройки ===
IMG_SIZE = (192, 256)

# 1. Трансформации
# === Функция предобработки изображения ===
def resize_and_center(img, size):
    """
    Масштабирует изображение с сохранением пропорций и добавляет черные поля
    для центрирования объекта
    """
    # Создаем копию, чтобы не изменять оригинал
    img = img.copy()

    # Определяем ориентацию (альбомная/портретная)
    is_landscape = img.width > img.height

    # Масштабируем по большей стороне
    if is_landscape:
        new_width = size[0]
        new_height = int(size[0] * img.height / img.width)
    else:
        new_height = size[1]
        new_width = int(size[1] * img.width / img.height)

    # Ресайз с сохранением пропорций
    img = img.resize((new_width, new_height), Image.BILINEAR)

    # Создаем новый холст с черным фоном
    new_img = Image.new("RGB", size, (0, 0, 0))

    # Центрируем изображение
    left = (size[0] - img.width) // 2
    top = (size[1] - img.height) // 2
    new_img.paste(img, (left, top))

    return new_img


# === Трансформации ===
transform = transforms.Compose([
    transforms.Lambda(lambda img: resize_and_center(img, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])


# === Загрузка метаданных ===
def load_metadata():
    """Загружает названия классов и описания из JSON-файлов"""
    try:
        with open('class_names.json', 'r', encoding='utf-8') as f:
            class_names = json.load(f)

        with open('class_descriptions.json', 'r', encoding='utf-8') as f:
            class_descriptions = json.load(f)

        return class_names, class_descriptions
    except FileNotFoundError:
        st.error("Ошибка: файлы метаданных не найдены!")
        st.stop()


CLASS_NAMES, CLASS_DESCRIPTIONS = load_metadata()

@st.cache_resource
def load_model():
    model = torch.jit.load("trashnet_mobile.pt", map_location='cpu')
    model.eval()
    return model

model = load_model()

# === Заголовок и описание ===
st.title("🗑️ ЭкоПомощник: Определение класса утилизации мусора")

# === Рекомендации для пользователя ===
st.subheader("📝 Как получить лучший результат?")
with st.expander("Показать советы по съемке"):
    st.markdown("""
    Чтобы повысить точность распознавания, следуйте этим рекомендациям:

    - **Контрастный фон**: Снимайте объект на однородном контрастном фоне
    - **Заполнение кадра**: Мусор должен занимать ≥70% кадра
    - **Освещение**: Избегайте бликов и сильных теней
    - **Мелкие объекты**: Используйте режим макросъемки
    - **Один объект**: Не фотографируйте несколько объектов одновременно
    - **Четкость**: Убедитесь, что объект в фокусе
    - **Угол съемки**: Снимайте объект прямо сверху или сбоку под углом 90°
    """)

# === Интерфейс ===
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
        probs = torch.softmax(output, dim=1)
        predicted_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_index].item()
        st.success(f"🧠 Предсказанный класс: **{description}** (доверие: {confidence:.2%})")


# === Социальная направленность проекта ===
st.divider()
st.subheader("♻️ О проекте")
st.markdown("""
**ЭкоПомощник** - социальный проект, созданный для повышения экологической сознательности и упрощения сортировки отходов. 
Наша миссия - сделать переработку мусора доступной и понятной для каждого.
""")

# Блок для пожертвований
st.markdown("### 💚 Поддержать проект")
col1, col2 = st.columns([1, 2])
with col1:
    st.info("QR-код для перевода")
with col2:
    st.markdown("""
    **Реквизиты для перевода:**
    - СБП: `+7 (XXX) XXX-XX-XX`
    - Карта: `XXXX XXXX XXXX XXXX`
    - ЮMoney: [https://yoomoney.ru/to/41001123456789](https://yoomoney.ru/to/41001123456789)

    *Любая сумма поможет развитию проекта!*
    """)

# === Контактная информация ===
st.divider()
st.markdown("""
**Контакты для сотрудничества:**
- 📧 Email: suhih.v@yandex.ru
- 📱 Telegram: @VJATCHESLAV87
""")
