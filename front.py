import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class NNLogic(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.first(x)
        return self.second(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNLogic()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

st.title("MNIST Classifier")
st.write("Загрузите изображение с цифрой, и модель попробует её распознать")

uploaded_file = st.file_uploader(
    "Выберите изображение",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", width=200)

    if st.button("Определить цифру"):
        img = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img).argmax(dim=1).item()

        st.success(f"Модель думает, что это цифра: {pred}")