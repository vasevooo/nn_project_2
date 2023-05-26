import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torch
from torchvision.models import resnet50
import torch.nn as nn
from torchvision import transforms


st.markdown("## Cat vs. Dog Classifier")



def main():
    
    st.text("Upload an image and the classifier will determine if it's a cat or a dog.")

    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Read the image file
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_tensor = preprocess_image(image)

        output = model(img_tensor.unsqueeze(0)).sigmoid().round()
       
        class_label = get_class_label(output.item())

        st.success(f"The uploaded image is a {class_label}.")

def get_class_label(class_label):
    if class_label == 0:
        return "cat"
    elif class_label == 1:
        return "dog"
    else:
        return "unknown"

def preprocess_image(image):
    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transforms_pipeline(image)

if __name__ == "__main__":
    # Load the pre-trained model
    
    model_url = 'https://github.com/vasevooo/nn_project/blob/main/resnet50_dogs_cats.pth'
    model_path = 'resnet50_dogs_cats.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = resnet50()
    model.fc = nn.Linear(2048, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    main()
