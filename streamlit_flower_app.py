import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
model = load_model('./flowers_vgg.h5')

classes = { 
    0:'its a daisy',
    1:'its a dandelion',
    2:'its a rose',
    3:'its a sunflower',
    4:'its a tulip',
 
}

st.title('Flower Classification')
st.header('Flower Classification Web App')
st.write('This is a flower classification app using Machine Learning')
uploaded_file = st.file_uploader("Choose a flower image file", type="jpg")

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR",width = 200)

btn = st.button("Flower Classifier")

if btn:
    image = np.expand_dims(opencv_image, axis=0)
    image = image/255
    pred = model.predict([image])
    st.write(pred)
    pred_class= np.argmax(pred)
    sign = classes[pred_class]
    #print(sign)
    st.write('Flower Classification : ' )
    st.success(sign)
