from PIL import Image

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from Config import *
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DATA_PATH = os.path.join('.', 'Dataset')

"""## **Class mapping**"""
class_to_idx = {
  'Speed Limit 10': 0,
  'Speed Limit 20': 1,
  'Speed Limit 30': 2,
  'Speed Limit 40': 3,
  'Speed Limit 50': 4,
  'Speed Limit 60': 5,
  'Speed Limit 70': 6,
  'Speed Limit 80': 7,
  'Speed Limit 90': 8,
  'Speed Limit 100': 9,
  'Speed Limit 110': 10,
  'Speed Limit 120': 11,
  'Stop': 12
}

class_names = {
    0: 'Speed Limit 10',
    1: 'Speed Limit 20',
    2: 'Speed Limit 30',
    3: 'Speed Limit 40',
    4: 'Speed Limit 50',
    5: 'Speed Limit 60',
    6: 'Speed Limit 70',
    7: 'Speed Limit 80',
    8: 'Speed Limit 90',
    9: 'Speed Limit 100',
    10: 'Speed Limit 110',
    11: 'Speed Limit 120',
    12: 'Stop'
}


"""## **Implement Normalization(Transform)**"""

def get_transform(is_training=True):
    base_transform = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Bilinear interpolation
        transforms.ToTensor()
    ]

    if is_training:
        train_transform = [
            # Tilt/Rotation: randomly rotate image within a range
            transforms.RandomRotation(degrees=15),

            # Blur: apply Gaussian blur with random kernel size
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.3),

            # Illumination changes: brightness, contrast, saturation, hue
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

            # We can add more augmentations here
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3),  # Translation
        ]

        # Insert augmentations before the base transforms
        transform = transforms.Compose(train_transform + base_transform)
    else:
        # No augmentation for validation/testing
        transform = transforms.Compose(base_transform)

    return transform


"""## **Define Loader**"""

class TrafficSignDataset(Dataset):
    def __init__(self, data_type, root_dir, transform=None, target_size=(IMAGE_SIZE, IMAGE_SIZE)):

        self.data_type = data_type
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        # Find all image files and their labels
        self.samples = []
        
        with open(os.path.join(BASE_DATA_PATH, self.data_type, '_annotations.csv'), 'r') as fh:
          lines = fh.readlines()
          for line in lines:
            if line.strip() != '' and not line.startswith('filename'):
              splitted = line.split(',')
              filename = splitted[0]
              label = splitted[3]
              self.samples.append((
                  os.path.join(self.root_dir, filename),
                  class_to_idx[label]
              ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.BILINEAR)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleTrafficSignNet(nn.Module):
    def __init__(self, num_classes=13):
        super(SimpleTrafficSignNet, self).__init__()

        # Layer definitions
        self.conv1 = nn.Conv2d(3, IMAGE_SIZE, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(IMAGE_SIZE, IMAGE_SIZE*2, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout(0.2)

        self.conv3 = nn.Conv2d(IMAGE_SIZE*2, IMAGE_SIZE*4, kernel_size=5, padding=2)
        self.dropout3 = nn.Dropout(0.3)

        self.pool4 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout4 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(28672, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x1 = self.conv1(x)               
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)
        x1_pooled = self.pool(x1)        

        x2 = self.conv2(x1_pooled)       
        x2 = F.relu(x2)
        x2 = self.dropout2(x2)
        x2_pooled = self.pool(x2)        

        x3 = self.conv3(x2_pooled)       
        x3 = F.relu(x3)
        x3 = self.dropout3(x3)
        x3_pooled = self.pool(x3)        

        path1 = self.pool4(x1)            
        path2 = self.pool5(x2_pooled)     
        path3 = x3_pooled                 

        flat1 = torch.flatten(path1, 1)   
        flat2 = torch.flatten(path2, 1)   
        flat3 = torch.flatten(path3, 1)   

        merged = torch.cat([flat1, flat2, flat3], dim=1)

        dense1 = self.fc1(merged)
        dense1 = F.relu(dense1)
        dense1 = self.dropout4(dense1)

        # Output layer (logits, no activation)
        output = self.fc2(dense1)

        return output



def infer(model, cv_image):
    # Set model to evaluation mode
    cv2_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(cv2_image_rgb)
    transform = transforms.ToTensor()

    image_tensor = transform(image)
    image_tensor = image_tensor.permute(1, 2, 0)

    image_tensor = image_tensor.to(device)
    model = model.to(device)

    return predict(model, image_tensor)

# This predict function will be also used by adversarial attacks!
def predict(model, x_in):
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        # Get current image dimensions
        h, w, c = x_in.shape

        # Convert numpy to tensor and add batch dimension
        x_tensor = torch.tensor(x_in, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        if torch.any(x_tensor > 1.0):
            x_tensor = x_tensor / 255.0

        # Resize to model input size if needed
        if h != IMAGE_SIZE or w != IMAGE_SIZE:
            x_tensor = F.interpolate(x_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

        outputs = model(x_tensor)

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prediction = outputs.argmax(dim=1).cpu().numpy()[0]


    return prediction, probabilities.squeeze().cpu().numpy()

def predict_class_only(model, x_in):
    prediction, _ = predict(model, x_in)
    return prediction

# Function to display prediction results
def display_prediction(predicted_class, probabilities):
    """
    Display the prediction results

    Args:
        predicted_class: Index of the predicted class
        probabilities: Probabilities for each class
    """
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]*100:.2f}%")

    # Display top 3 predictions
    top3_indices = np.argsort(probabilities)[-3:][::-1]

    plt.figure(figsize=(10, 5))
    plt.barh([class_names[i] for i in top3_indices],
             [probabilities[i] for i in top3_indices])
    plt.xlabel('Probability')
    plt.title('Top 3 Predictions')
    plt.tight_layout()
    plt.show()

def perform_inference(model, cv_image, should_crop=False, display_pred=False):
    if should_crop:
      cv_image = crop_with_custom_model_cv2(cv_image)
      if cv_image is None:
          print("Nothing detected.")
          return None, None

    # Perform inference
    predicted_class, probabilities = infer(model, cv_image)

    if display_pred:
      display_prediction(predicted_class, probabilities)

    return predicted_class, probabilities


def load_model(model_path, num_classes=13):
    model = SimpleTrafficSignNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


