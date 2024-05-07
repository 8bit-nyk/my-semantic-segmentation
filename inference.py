# inference.py
import torch
from model import get_model
import numpy as np
import matplotlib.pyplot as plt
from data_preparation import get_transform
from torchvision import transforms as T
import cv2

def load_image(image_path):
    # Use OpenCV to read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    # Apply transformations
    transform = get_transform(False)  # Ensure no augmentation is applied
    augmented = transform(image=image)
    image_tensor = augmented['image']  # Already a tensor
    return image_tensor.unsqueeze(0)  # Add batch dimension


def visualize_prediction(original_image, mask):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Convert boolean mask to uint8 before resizing
    mask = mask.squeeze(0)  # Remove batch dimension
    mask_uint8 = mask.astype(np.uint8) * 255  # Convert boolean to uint8 and scale to 0-255
    
    # Resize mask back to the original image size for accurate overlay
    resized_mask = cv2.resize(mask_uint8, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    plt.imshow(resized_mask, cmap='jet', alpha=0.5)  # Using a colormap for better visibility
    plt.title('Prediction Overlay')
    plt.axis('off')
    
    plt.show()

def run_inference(model_path, image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    image_tensor = load_image(image_path).to(device)
    
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = torch.sigmoid(prediction).squeeze(0)  # Remove batch dim for visualization
        predicted_mask = (prediction > 0.5).cpu().numpy()

    # Load original image for visualization
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    visualize_prediction(original_image, predicted_mask)

if __name__ == "__main__":
    model_path = 'models/binary_model_Unetpp.pth'  # Adjust the path as needed
    image_path = 'inputs/syn_iwhub_4.png'

    #image_path = '/home/aub/datasets/idealworks_poi_syntetic_dataset/IdealWorks dataset/Charger_POI/RGB Images/rgb_0141.png'
# Path to an image to test
    run_inference(model_path, image_path)