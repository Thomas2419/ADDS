import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms

# Function to extract all possible labels from the JSON structure
def extract_labels(json_data):
    labels = set()

    def recursive_extract(d):
        for key, value in d.items():
            if isinstance(value, list):
                if value:
                    labels.add(key)
                    for item in value:
                        labels.add(item)
                else:
                    labels.add(key)
            elif isinstance(value, dict):
                labels.add(key)
                recursive_extract(value)

    recursive_extract(json_data)
    return sorted(labels)

# Function to preprocess JSON files to add multi-hot encoded labels
def preprocess_json_files(data_dir, possible_labels):
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        label_file = os.path.splitext(image_file)[0] + '.json'
        label_path = os.path.join(data_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file not found for image: {image_file}")
            continue

        with open(label_path, 'r') as f:
            label_data = json.load(f)

        # Collect labels from the JSON structure
        labels = set()
        def recursive_collect(d):
            for key, value in d.items():
                if isinstance(value, list):
                    if value:
                        labels.add(key)
                        labels.update(value)
                    else:
                        labels.add(key)
                elif isinstance(value, dict):
                    labels.add(key)
                    recursive_collect(value)

        recursive_collect(label_data)

        # Convert the set of labels to a multi-hot encoded vector
        encoded_labels = [1 if label in labels else 0 for label in possible_labels]

        # Add the multi-hot encoded labels back into the JSON file
        label_data["encoded_labels"] = encoded_labels

        # Save the updated JSON file
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=4)

    print("Preprocessing complete.")

class JSONLabelDataset(Dataset):
    def __init__(self, data_dir, image_size=896):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()  # Use torchvision's ToTensor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_file)

        image = Image.open(image_path)

        # Handle P and RGBA images by pasting them onto a white background
        if image.mode == 'P':
            image = image.convert("RGBA")
        if image.mode == 'RGBA':
            # Create a white background
            background = Image.new("RGB", image.size, (255, 255, 255))
            # Paste the RGBA image onto the white background, using the alpha channel as a mask
            background.paste(image, mask=image.split()[3])  # Use alpha channel as mask
            image = background

        # Convert other modes directly to RGB
        else:
            image = image.convert("RGB")

        # Make the image square by pasting it onto a white square background
        max_side = max(image.size)
        square_background = Image.new("RGB", (max_side, max_side), (255, 255, 255))
        square_background.paste(image, (int((max_side - image.width) / 2), int((max_side - image.height) / 2)))
        
        # Resize the image to the specified size (896x896)
        image = square_background.resize((self.image_size, self.image_size))

        # Convert image to tensor
        image = self.to_tensor(image)

        label_file = os.path.splitext(image_file)[0] + '.json'
        label_path = os.path.join(self.data_dir, label_file)

        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        labels = torch.tensor(label_data["encoded_labels"], dtype=torch.float32)

        return image, labels


