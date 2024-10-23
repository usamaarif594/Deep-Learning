import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# Load tokenizer
with open('tokenizer_.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load captioning model
model = load_model('best_model.h5')

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length=35):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    # Remove startseq and endseq from the generated caption
    final_caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    
    return final_caption


# Function to convert index to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Streamlit UI
st.title('Captionify: AI Image Caption Generator')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=300)

    # Load and preprocess the image for VGG16
    img = load_img(uploaded_file, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

    # Extract features using VGG16
    vgg_model = VGG16()
    vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
    feature = vgg_model.predict(img_array, verbose=1)

    if st.button('Generate Caption'):
        caption = predict_caption(model, feature, tokenizer)
        st.success(f'Generated Caption: {caption}')
        # Footer
        st.text("____")



