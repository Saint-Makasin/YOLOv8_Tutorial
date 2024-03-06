import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

st.title("...")
model = YOLO('path')
images = st.file_uploader("...", accept_multiple_files=True)

for image in images:
    img = Image.open(image)
    predicted_class = model.predict(img, device=device)[0]
    predicted_class.save(image.name)
    filename = f'{image.name[:-4]}.txt'
    open(filename, 'a')
    predicted_class.save_txt(filename)
    st.image(image.name, caption='image')
    st.download_button(label='...', data=open(filename, 'r'), file_name=filename)

