import streamlit as st
from PIL import Image
import requests

api = 'http://127.0.0.1:8000/predict/'

st.title('MNIST Model')
st.write('Загрузите изображение с цифрой и модель попробует распознать')

uploaded_file = st.file_uploader('Выберите изображение', type=['png, jpg, jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение', width=200)

    if st.button('Определить'):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            request = requests.post(api, files=files)

            if request.status_code == 200:
                result = request.json()
                st.success(f'Модель думает, что это цифра: {result['Prediction']}')
            else:
                st.error(f'Error: {request.status_code}')
        except requests.exceptions.RequestException:
            st.error('Cannot connect to the API')