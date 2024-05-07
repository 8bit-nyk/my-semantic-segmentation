
import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split

class CustomDataset(Dataset):
    def __init__(self, data_root, transforms=None, subset_size=None, subset_indices=None):
        self.data_root = data_root
        self.transforms = transforms
        image_paths = sorted([os.path.join(data_root, 'images', fname) 
                              for fname in os.listdir(os.path.join(data_root, 'images'))])
        mask_paths = sorted([os.path.join(data_root, 'masks', fname) 
                             for fname in os.listdir(os.path.join(data_root, 'masks'))])
        label_paths = sorted([os.path.join(data_root, 'labels', fname) 
                              for fname in os.listdir(os.path.join(data_root, 'labels'))])

        # If specific indices are provided, use them to subset the dataset
        if subset_indices is not None:
            self.images = [image_paths[i] for i in subset_indices]
            self.masks = [mask_paths[i] for i in subset_indices]
            self.labels = [label_paths[i] for i in subset_indices]
        elif subset_size is not None:
            # Subsample the dataset by a specified fraction
            sample_indices = random.sample(range(len(image_paths)), int(len(image_paths) * subset_size))
            self.images = [image_paths[i] for i in sample_indices]
            self.masks = [mask_paths[i] for i in sample_indices]
            self.labels = [label_paths[i] for i in sample_indices]
        else:
            self.images = image_paths
            self.masks = mask_paths
            self.labels = label_paths

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load images and masks
        image_path = self.images[idx]
        mask_path = self.masks[idx]
        label_path = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        with open(label_path) as f:
            labels = json.load(f)
        # #Binary classification:
        mask = Image.open(mask_path).convert("RGB")

        #'iwhub' is the class of interest and its color is [140, 25, 255]
        binary_mask = convert_rgb_to_binary_mask(mask, [140, 25, 255])
        class_mask = binary_mask

        # #Multi-class classification:
        # mask = Image.open(mask_path).convert("RGBA")  # Ensure mask is loaded with alpha
        # mask_array = np.array(mask)
        # # Initialize an array to hold class indices
        # class_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.int64)
        # # Convert RGBA to class indices
        # for color, info in labels.items():
        #     rgba = eval(color)  # Convert color string to tuple
        #     class_idx = self._class_to_index(info['class'])
        #     mask_match = (mask_array[:, :, :3] == rgba[:3]).all(axis=-1)  # Match only RGB, ignore alpha
        #     class_mask[mask_match] = class_idx

        # Apply transformations
        if self.transforms:
            augmented = self.transforms(image=np.array(image), mask=class_mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

def _class_to_index(self, class_name):
    # Convert class name to index
    class_to_idx = {'BACKGROUND': 0, 'boxes': 1, 'UNLABELLED': 2, 'iwhub': 3, 'racks': 4,
                    'dolly': 5, 'forklift': 6, 'stillage': 7, 'boxes,pallet': 8, 'pallet': 9, 'railing': 10}
    return class_to_idx.get(class_name, -1)  # Default to -1 for unknown classes
    
def convert_rgb_to_binary_mask(mask, target_color):
    #Convert an RGB mask to binary format. Pixels matching the target color will be 1, others will be 0
    mask_array = np.array(mask)
    target_color = np.array(target_color)
    binary_mask = (mask_array == target_color[:3]).all(axis=-1).astype(np.float32)
    return binary_mask

def get_transform(augment):
    if augment:
        return A.Compose([
            A.Resize(height=256, width=256),  # Resize to nearest dimensions divisible by 32
            #A.RandomCrop(width=256, height=256),
            #A.Resize(height=128, width=128), # Resize for quicker training
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=256, width=256),  # Resize to meet dimension requirement
            #A.Resize(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# Use this function to get a data loader
def get_loader(data_root, batch_size, augment=False,shuffle=True, subset_size=None,subset_indices=None):
    dataset = CustomDataset(data_root, get_transform(augment),subset_size=subset_size,subset_indices=subset_indices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# Train/Val/Test split function
def create_data_splits(total_items, test_ratio=0.2, validation_ratio=0.25):
    #image_paths = sorted([os.path.join(data_root, 'images', fname) 
     #                     for fname in os.listdir(os.path.join(data_root, 'images'))])
    # Generate indices and split them
    #indices = list(range(len(image_paths)))
    indices = list(range(total_items))
    train_indices, test_indices = train_test_split(indices, test_size=test_ratio, random_state=42)
    # Adjust validation ratio to compensate for reduced training set
    train_indices, val_indices = train_test_split(train_indices, test_size=validation_ratio, random_state=42)
    return train_indices, val_indices, test_indices

if __name__ == '__main__':
    # Paths
    data_root = 'data'  # Replace with the path to your data directory
    batch_size = 4 # Can change this based on how much memory you have
    # Split the data
    train_indices, val_indices, test_indices = create_data_splits(1000)
    print(f"Train size: {len(train_indices)}, Val size: {len(val_indices)}, Test size: {len(test_indices)}")
    print("Train Indices Sample:", train_indices[:5])  # Print first 5 indices of training set
    print("Validation Indices Sample:", val_indices[:5])  # Print first 5 indices of validation set
    print("Test Indices Sample:", test_indices[:5])  # Print first 5 indices of test set
    # Get the data loader
    data_loader = get_loader(data_root, batch_size, augment=True,subset_size=0.1)
    print(f"dataloader subset:{ data_loader.__len__()}")
    # train_loader = get_loader('data', batch_size, True, subset_indices=train_indices)
    # val_loader = get_loader('data', batch_size, False, subset_indices=val_indices)
    # test_loader = get_loader('data', batch_size, False, subset_indices=test_indices)
    # print(f"dataloader train: {train_loader.__len__()}")
    # print(f"dataloader val:{val_loader.__len__()}")
    # print(f"dataloader test:{test_loader.__len__()}")
    # Quick check to see if the data loader works
    for images, masks in data_loader:
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        break  # Just check the first batch

# Displaying a batch of images and masks to verify augmentations
  # Fetch a single batch from the data loader
    batch = next(iter(data_loader))
    images, masks = batch
    
    # Set up the figure
    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))  # Three rows for images, masks, and overlays

    for i in range(4):
        # Display images
        ax[0, i].imshow(images[i].permute(1, 2, 0))  # Images row
        ax[0, i].set_title(f'Image {i}')
        ax[0, i].axis('off')

        # Display masks
        ax[1, i].imshow(masks[i], cmap='gray')  # Masks row
        ax[1, i].set_title(f'Mask {i}')
        ax[1, i].axis('off')

        # Display overlay of images and masks
        ax[2, i].imshow(images[i].permute(1, 2, 0))
        ax[2, i].imshow(masks[i], alpha=0.5, cmap='jet')  # Overlay row
        ax[2, i].set_title(f'Overlay {i}')
        ax[2, i].axis('off')

    plt.tight_layout()
    plt.show()
