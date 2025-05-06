# -*- coding: utf-8 -*-
import locale
locale.getpreferredencoding = "UTF-8"

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import os
import time
import random

from Model import SimpleTrafficSignNet, get_transform, class_to_idx, class_names, TrafficSignDataset
from Config import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Visualization functions
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


# Evaluate model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    print("Evaluating model on test data...")
    correct_by_class = {}
    total_by_class = {}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Track per-class accuracy
            for pred, label in zip(preds, labels):
                label_item = label.item()
                if label_item not in total_by_class:
                    total_by_class[label_item] = 0
                    correct_by_class[label_item] = 0

                total_by_class[label_item] += 1
                if pred == label:
                    correct_by_class[label_item] += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar with current accuracy
            batch_acc = torch.sum(preds == labels).double() / len(labels)
            pbar.set_postfix(acc=f'{batch_acc:.4f}')

    # Overall accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Test Accuracy: {accuracy:.4f}')

    # Per-class accuracy
    print("\nPer-class accuracy:")
    print("-" * 30)
    print(f"{'Class':^10} | {'Accuracy':^10} | {'Samples':^10}")
    print("-" * 30)

    for class_idx in sorted(total_by_class.keys()):
        class_acc = correct_by_class[class_idx] / total_by_class[class_idx]
        print(f"{class_idx:^10} | {class_acc:.4f} | {total_by_class[class_idx]:^10}")

    return all_preds, all_labels


# training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, scheduler=None):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Print model information at the start
    print(f'Model Architecture:')
    print(model)
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print('-' * 50)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Back Prop.. and optimize
            loss.backward()
            optimizer.step()

            # Calculate batch accuracy
            batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            batch_count += 1

            # Update progress bar
            pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}')

            # Print detailed batch statistics every 10 batches
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: '
                      f'Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0

        # Add progress bar for validation
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Calculate batch accuracy
                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{batch_acc:.4f}')

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc.item())

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Update learning rate if using ReduceLROnPlateau scheduler
        if scheduler and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_acc)
        elif scheduler:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
            print(f'Learning rate: {current_lr:.6f}')

        # Save the best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_traffic_sign_model.pth')
            print(f'New best model saved! Validation accuracy: {epoch_acc:.4f}')

    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(torch.load('best_traffic_sign_model.pth'))

    return model, train_losses, val_losses, train_accs, val_accs


# Run the full training pipeline
def start_train():
    # Parameters
    batch_size = BATCH_SIZE
    num_epochs = NUM_EPOCHS
    num_classes = NUM_CLASSES
    learning_rate = LEARNING_RATE

    print(f"Training Configuration:")
    print(f"----------------------")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Classes: {num_classes}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"----------------------")

    # Data directories
    data_root = os.path.join('.', 'Dataset')

    # Get transforms
    train_transform = val_transform = get_transform()

    # Create datasets
    train_dataset = TrafficSignDataset(
        'train',
        root_dir=os.path.join(data_root, 'train'),
        transform=train_transform
    )

    val_dataset = TrafficSignDataset(
        'valid',
        root_dir=os.path.join(data_root, 'valid'),
        transform=val_transform
    )

    test_dataset = TrafficSignDataset(
        'test',
        root_dir=os.path.join(data_root, 'test'),
        transform=val_transform
    )

    print(f"\nDataset Information:")
    print(f"-------------------")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Class mapping: {class_to_idx}")
    print(f"-------------------")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SimpleTrafficSignNet(num_classes=num_classes)
    model = model.to(device)

    # Loss function and optimizer with weight decay
    LR = 0.0001                                     # Learning rate
    L2_LAMBDA = 0.0001                              # Lambda for l2 regularization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2_LAMBDA)

    # Learning rate scheduler - use ReduceLROnPlateau for adaptive adjustment
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    # Train model
    print("\nStarting training...")
    print("=" * 50)
    start_time = time.time()

    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, scheduler=scheduler
    )

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print("=" * 50)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Evaluate on test set
    evaluate_model(model, test_loader)

    print("Training completed!")

"""## **(Inference) Test the model**"""
def load_model(model_path, num_classes=13):
    model = SimpleTrafficSignNet(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


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

def crop_with_custom_model_cv2(cv_image):
  # Will be implemented later!
  pass

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

def test_sample():
    img = cv2.imread("sample.png")
    model = load_model('best_traffic_sign_model.pth')
    predicted_class, probabilities = perform_inference(model, img, should_crop=False, display_pred=True)

    if predicted_class is not None and probabilities is not None:
      print(f'prediction : {class_names[predicted_class]}', ',' , f'conf: {probabilities[predicted_class]*100} %')

def main():
    print(f"Using device: {device}")
    start_train()
    #test_sample()

if __name__ == '__main__':
    main()