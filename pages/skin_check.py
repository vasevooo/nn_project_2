import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
from torchvision.models import resnet50
import streamlit as st
from torchvision import transforms

def main():
    st.title("Skin Cancer Classification")
    st.write("Skin Cancer Classification is a vital task in medical image analysis.It involves the identification and categorization of skin lesions into malignant and benign classes using machine learning techniques.") 

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        img_tensor = preprocess_image(image)

        output = model(img_tensor.unsqueeze(0)).sigmoid().round()
       
        class_label = get_class_label(output.item())

        st.success(f"The uploaded image is a {class_label}.")


def preprocess_image(image):
    transforms_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transforms_pipeline(image)

def get_class_label(class_label):    
    if class_label == 0:
        return 'Benign'
    elif class_label == 1:
        return 'Malignant'

if __name__ == "__main__":
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet50(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=1)

        model_path = 'app_skin_care/resnet50bin2.pth'

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        main()


