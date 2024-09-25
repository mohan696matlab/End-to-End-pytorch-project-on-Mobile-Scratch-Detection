import streamlit as st
from infer import predict_screen_defect

st.title('Mobile Scratch Detection')
st.markdown('---')

image = st.file_uploader("upload screen image",type=['png','jpg'] )

if image is not None:
    st.image(image, caption='uploaded image')

    pred,confidence = predict_screen_defect(image)
    st.markdown(f'## Predicted class: {pred}, confidence: {confidence:.3f}')