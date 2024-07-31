import streamlit as st
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm.auto import tqdm
from torch import nn
import torchvision
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torchmetrics
import torchinfo

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.title("Mobile Scratch Detection")
st.markdown("---")
st.markdown(f"### Model is running on : {device}")

classes = ['oil', 'scratch', 'stain']
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])

st.write("Loading the pretrained model...\n")

cnn_model = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights)
for param in cnn_model.parameters():
    param.requires_grad = False
cnn_model.fc = nn.Linear(2048,out_features=3)
cnn_model.load_state_dict(torch.load('cnn_model_weights.pth'))
cnn_model.to(device)
cnn_model.eval()

st.write("Pretrained model loaded sucessfully...\n")





image = st.file_uploader("Upload the image",type=['png', 'jpg', 'jpeg'])


if image is not None:

    st.image(image, caption="Uploaded image")

    image = Image.open(image)
    
    image = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        y_pred = cnn_model(image)
    predicted_classes = [classes[yy] for yy in y_pred.argmax(dim=-1).cpu()]
    confidence = torch.nn.functional.softmax(y_pred.cpu()).max()
    st.markdown(f"## Predicted class: {predicted_classes[0]} ({confidence*100:0.4f} %)")
    # print(y_pred)


    # fig=plt.figure()
    # plt.imshow(image)
    # st.pyplot(fig)