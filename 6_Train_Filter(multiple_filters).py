# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image
from tqdm import tqdm

import os
import time
import random

from Model import *
from Config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model('best_traffic_sign_model.pth')

"""## **Train a Filter (with N*N sized kernel)**"""
FILTER_SIZE_1 = 7
FILTER_SIZE_2 = 5
low  = -0.1
high = 2.0


class AdversarialFilterTrainer:
    def __init__(self, model_path, initial_filter=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleTrafficSignNet(num_classes=13)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # Set to evaluation mode
        self.model.to(self.device)

        # Freeze the classifier model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        filter_weights = (high - low) * torch.rand((FILTER_SIZE_1, FILTER_SIZE_1)) + low
        filter_weights = filter_weights / filter_weights.sum()
        initial_filter1 = filter_weights.unsqueeze(0).unsqueeze(0) # Convert to PyTorch parameter with correct shape for convolution
        initial_filter1 = initial_filter1.repeat(3, 1, 1, 1) # Expand to match input channels (3 for RGB)
        self.filter1 = nn.Parameter(initial_filter1.to(self.device)) # Create parameter and move to the same device as the model

        filter_weights = (high - low) * torch.rand((FILTER_SIZE_2, FILTER_SIZE_2)) + low
        filter_weights = filter_weights / filter_weights.sum()
        initial_filter2 = filter_weights.unsqueeze(0).unsqueeze(0) # Convert to PyTorch parameter with correct shape for convolution
        initial_filter2 = initial_filter2.repeat(3, 1, 1, 1) # Expand to match input channels (3 for RGB)
        self.filter2 = nn.Parameter(initial_filter2.to(self.device)) # Create parameter and move to the same device as the model

        # Define loss function and optimizer with higher learning rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam([self.filter1, self.filter2], lr=0.01)  # Increased from 0.001

    def apply_filter(self, images):
        # Ensure images are on the right device
        images = images.to(self.device)

        # Store original mean intensity per image and channel
        original_mean = images.mean(dim=[2, 3], keepdim=True)

        # Set padding for both filters
        padding1 = self.filter1.shape[2] // 2
        padding2 = self.filter2.shape[2] // 2

        # Apply first filter
        filtered_images = nn.functional.conv2d(
            images,
            self.filter1,
            padding=padding1,
            groups=3  # Apply the filter to each channel separately
        )

        # Apply second filter
        filtered_images = nn.functional.conv2d(
            filtered_images,
            self.filter2,
            padding=padding2,
            groups=3  # Apply the filter to each channel separately
        )

        # Calculate filtered image mean intensity
        filtered_mean = filtered_images.mean(dim=[2, 3], keepdim=True)

        # Adjust brightness to match original image mean intensity
        # Add small epsilon to avoid division by zero
        brightness_ratio = original_mean / (filtered_mean + 1e-10)
        filtered_images = filtered_images * brightness_ratio

        filtered_images = torch.clamp(filtered_images, 0, 255)

        return filtered_images

    def train_step(self, adv_images, orig_labels, adv_labels):
        self.optimizer.zero_grad()

        # Get batch size
        batch_size = adv_images.size(0)

        # Apply the filter to the adversarial images
        filtered_images = self.apply_filter(adv_images)

        # Get model predictions
        logits = self.model(filtered_images)
        probs = torch.softmax(logits, dim=1)

        # Move labels to device if needed
        orig_labels = orig_labels.to(self.device)

        # Gather probabilities for both original and adversarial labels
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # For original labels
        orig_probs = probs[batch_indices, orig_labels]
        
        # For adversarial labels
        adv_probs = probs[batch_indices, adv_labels]

        # Store probabilities for return
        recovered_probs = orig_probs.detach().cpu().numpy().tolist()

        # 1. Direct maximization of target class probabilities
        direct_prob_loss = -torch.log(orig_probs).mean()

        # 2. Direct minimization of adversarial class probabilities
        # Add a weight parameter to control the importance of this loss
        adv_prob_weight = 1.0  # Adjust as needed
        adv_prob_loss = torch.log(adv_probs).mean() * adv_prob_weight

        # 3. Classification cross-entropy loss
        ce_loss = self.criterion(logits, orig_labels)

        # 4. Margin loss - maximize gap between original and adversarial
        margin_loss = torch.clamp(adv_probs - orig_probs + 0.2, min=0.0).mean()

        # 5. Entropy loss to prevent overconfidence
        entropy_loss = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        entropy_weight = 0.5

        # Combined loss
        loss = ce_loss + direct_prob_loss + adv_prob_loss + margin_loss + entropy_weight * entropy_loss

        # Backpropagate and update filter parameters
        loss.backward()
        self.optimizer.step()

        # Force filter values to be non-negative and normalize
        with torch.no_grad():
            self.filter1.data.clamp_(low, high)
            self.filter2.data.clamp_(low, high)
            #self.filter1.data = self.filter1.data / self.filter1.data.sum()
            #self.filter2.data = self.filter2.data / self.filter2.data.sum()

        return loss.item(), recovered_probs

    def get_filter_numpy(self):
        """Get the trained filter as a numpy array"""
        with torch.no_grad():
            # Average across input channels to get a single 2D filter
            # This gives us a filter we can use with cv2.filter2D
            filter1_np = self.filter1.data.mean(dim=0).squeeze().cpu().numpy()
            filter2_np = self.filter2.data.mean(dim=0).squeeze().cpu().numpy()
            return filter1_np, filter2_np

    def save_filter(self, best):
        """Save the filter to a numpy file"""
        filter1_np, filter2_np = self.get_filter_numpy()
        if best:
            np.save('best_filter_1.npy', filter1_np)
            np.save('best_filter_2.npy', filter2_np)
        else:
            np.save('filter_1.npy', filter1_np)
            np.save('filter_2.npy', filter2_np)
        return filter1_np, filter2_np


class AdversarialDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Dataset for adversarial examples

        Args:
            csv_file: Path to the CSV file with annotations
            transform: Optional transform to be applied to images
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get file paths and labels
        adv_path = self.data.iloc[idx]['ADV_FILEPATH']
        orig_predicted_label = self.data.iloc[idx]['ORIG_PREDICTED_LABEL']
        adv_predicted_label = self.data.iloc[idx]['ADV_PREDICTED_LABEL']

        # Load adversarial image
        adv_image = Image.open(adv_path).convert('RGB')

        # Apply transformations
        if self.transform:
            adv_image = self.transform(adv_image)

        return adv_image, orig_predicted_label, adv_predicted_label


def train():
    # Set up data loading
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize to match model input size
        transforms.ToTensor(),
    ])

    # Define batch size
    # batch_size = 700
    batch_size = 10

    # Create dataset and dataloader
    pd_dataset = pd.read_csv('adv_samples_printed_and_pictured/_annotations.csv')

    # Filter out JSMA and CW
    mask = pd_dataset['ATTACK_TYPE'].str.contains('jsma')
    selected_samples = pd_dataset[~mask].sort_values(by=['ORIGINAL_FILEPATH'])

    mask = selected_samples['ATTACK_TYPE'].str.contains('cw')
    selected_samples = selected_samples[~mask].sort_values(by=['ORIGINAL_FILEPATH'])

    mask = selected_samples['ATTACK_TYPE'].str.contains('fg')
    selected_samples = selected_samples[~mask].sort_values(by=['ORIGINAL_FILEPATH'])

    #selected_samples = pd_dataset

    train_size = int(0.9 * len(selected_samples))
    test_size = len(selected_samples) - train_size
    train_dataset, test_dataset = random_split(selected_samples, [train_size, test_size])

    test_indices = test_dataset.indices
    test_df = selected_samples.iloc[test_indices]
    test_df.to_csv('filter_test_dataset.csv', index=False)

    train_indices = train_dataset.indices
    train_df = selected_samples.iloc[train_indices]

    dataset = AdversarialDataset(
        data=train_df,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize filter trainer with identity + noise initialization
    trainer = AdversarialFilterTrainer(
        model_path='best_traffic_sign_model.pth'
    )

    # Training loop
    num_epochs = 50
    best_avg_recovery = 0.0
    patience = 10  # Early stopping patience
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_recoveries = []

        for adv_images, orig_labels, adv_labels in dataloader:
            # Try several optimizer steps for each batch to push harder
            for _ in range(3):  # Multiple optimization steps per batch
                loss, recovered_probs = trainer.train_step(adv_images, orig_labels, adv_labels)

            epoch_losses.append(loss)
            # Add all recoveries from the batch to the list
            epoch_recoveries.extend(recovered_probs)

        # Calculate average metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_recovery = sum(epoch_recoveries) / len(epoch_recoveries)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Avg Recovery Prob: {avg_recovery:.4f}")

        trainer.save_filter(best=False)
        # Save best filter
        if avg_recovery > best_avg_recovery:
            best_avg_recovery = avg_recovery
            trainer.save_filter(best=True)
            print(f"New best filter saved with recovery probability: {avg_recovery:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        """
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs without improvement")
            break
        """

        # Learning rate schedule - reduce learning rate if no improvement
        if epochs_no_improve > 0 and epochs_no_improve % 10 == 0:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"Reduced learning rate to {param_group['lr']}")


    # Print the final trained filter
    best_filter_1, best_filter_2 = trainer.get_filter_numpy()
    print(best_filter_1)
    print(best_filter_2)

"""## **Define BGR2RGB**"""
def rgb(cv_image):
  temp = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
  return temp


def apply_filter():
    """## APPLY A FILTER!** (Programmatically)"""
    filter1 = np.load('best_filter_1.npy')
    filter2 = np.load('best_filter_2.npy')

    df = pd.read_csv("filter_test_dataset.csv", header=0)

    images = [11]
    for image in images:
        mask = df['ATTACK_TYPE'].str.contains('spgd')
        selected_sample = df[mask].sort_values(by=['ORIGINAL_FILEPATH']).iloc[image]

        sample_orig_path = selected_sample.iloc[0]
        sample_orig_cv2_image = cv2.imread(sample_orig_path)
        sample_orig_pred_label = selected_sample.iloc[5]
        sample_orig_pred_prob = selected_sample.iloc[6]

        sample_orig_cv2_filtered_image = sample_orig_cv2_image.copy()
        original_mean = np.mean(sample_orig_cv2_filtered_image, axis=(0, 1), keepdims=True)
        orig_filtered_image = cv2.filter2D(sample_orig_cv2_filtered_image, -1, filter1, borderType=cv2.BORDER_REFLECT)
        orig_filtered_image = cv2.filter2D(orig_filtered_image, -1, filter2, borderType=cv2.BORDER_REFLECT)
        filtered_mean = np.mean(orig_filtered_image, axis=(0, 1), keepdims=True)
        brightness_ratio = original_mean / (filtered_mean + 1e-10)
        orig_filtered_image = orig_filtered_image * brightness_ratio
        orig_filtered_image = np.clip(orig_filtered_image, 0, 255).astype(np.uint8)
        orig_filtered_label, orig_filtered_prob = infer(model, orig_filtered_image)


        sample_adv_path = selected_sample.iloc[1]
        sample_adv_cv2_image = cv2.imread(sample_adv_path)

        original_mean = np.mean(sample_adv_cv2_image, axis=(0, 1), keepdims=True)
        filtered_image = cv2.filter2D(sample_adv_cv2_image, -1, filter1, borderType=cv2.BORDER_REFLECT)
        filtered_image = cv2.filter2D(filtered_image, -1, filter2, borderType=cv2.BORDER_REFLECT)
        filtered_mean = np.mean(filtered_image, axis=(0, 1), keepdims=True)
        brightness_ratio = original_mean / (filtered_mean + 1e-10)
        filtered_image = filtered_image * brightness_ratio
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        sample_filtered_pred_label, sample_filtered_pred_prob = infer(model, filtered_image)

        plt.rcParams.update({'font.size': 7})
        fig, axes = plt.subplots(1, 4)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
          ax.set_xticks([])  # Remove x-axis ticks
          ax.set_yticks([])  # Remove y-axis ticks

        # Display each image in a subplot

        axes[0].imshow(rgb(sample_orig_cv2_image), interpolation='nearest')
        axes[0].set_title(f'Original Image\n-\n-\n{class_names[selected_sample.iloc[5]]} ({selected_sample.iloc[6] * 100:.2f}%)')

        axes[1].imshow(rgb(orig_filtered_image), interpolation='nearest')
        axes[1].set_title(f'Original Image\nFilter applied\n-\n{class_names[orig_filtered_label]}({orig_filtered_prob[orig_filtered_label] * 100:.2f}%)')

        axes[2].imshow(rgb(sample_adv_cv2_image))
        axes[2].set_title(f'Attack Type: {selected_sample.iloc[2]}\nNo filter\n{class_names[selected_sample.iloc[7]]} ({selected_sample.iloc[8] * 100:.2f}%)\n{class_names[sample_orig_pred_label]} ({selected_sample.iloc[9] * 100:.2f}%)')

        axes[3].imshow(rgb(filtered_image))
        axes[3].set_title(f'Attack Type: {selected_sample.iloc[2]}\nFilter applied\n{class_names[sample_filtered_pred_label]} ({sample_filtered_pred_prob[sample_filtered_pred_label] * 100:.2f}%)\n{class_names[sample_orig_pred_label]} ({sample_filtered_pred_prob[sample_orig_pred_label] * 100:.2f}%)')

        plt.tight_layout()
        plt.show()
        print('\n\n')

def main():
    train()
    #apply_filter()

if __name__ == '__main__':
    main()
