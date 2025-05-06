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

from Model import *
from Config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ADV_SAMPLES_PATH = os.path.join('.', 'adv_samples')

# Visualize the original and adversarial images
def visualize_attack(test, original_image, adv_image, target_class, src_class, dst_class):
    """
    Visualizes the original and adversarial images side by side.
    """
    # Move tensors to CPU for visualization
    original_image_cpu = original_image.cpu()
    adv_image_cpu = adv_image.cpu()

    orig_img = original_image_cpu.squeeze(0)
    adv_img = adv_image_cpu.squeeze(0)

    # Calculate perturbation
    perturbation = orig_img - adv_img

    if test:
      # Calculate L0, L2, and L∞ norms
      l0_norm = torch.sum(perturbation != 0).item() / perturbation.numel()
      l2_norm = torch.norm(perturbation).item()
      linf_norm = torch.max(torch.abs(perturbation)).item()

      # Calculate perturbation percentage
      total_pixels = perturbation.shape[1] * perturbation.shape[2]
      nonzero_pixels = torch.sum(torch.sum(torch.abs(perturbation), dim=0) > 0).item()
      pixel_percent = (nonzero_pixels / total_pixels) * 100

      print(f"Perturbation stats:")
      print(f"- L0 norm (sparsity): {l0_norm:.4f} ({pixel_percent:.2f}% of pixels modified)")
      print(f"- L2 norm (magnitude): {l2_norm:.4f}")
      print(f"- L∞ norm (max change): {linf_norm:.4f}")

      # Normalize perturbation for better visibility
      perturbation_display = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-8)

      # Create figure
      fig, axs = plt.subplots(1, 3, figsize=(18, 6))

      # Original image
      axs[0].imshow(orig_img.permute(1, 2, 0).clamp(0, 1).numpy())
      axs[0].set_title(f"Original:\nPredicted={class_names[src_class]}")
      axs[0].axis('off')

      # Adversarial image
      if target_class == None:
        axs[1].imshow(adv_img.permute(1, 2, 0).clamp(0, 1).numpy())
        axs[1].set_title(f"Adversarial:\nPredicted={class_names[dst_class]}")
        axs[1].axis('off')
      else:
        axs[1].imshow(adv_img.permute(1, 2, 0).clamp(0, 1).numpy())
        axs[1].set_title(f"Adversarial:\nPredicted={class_names[dst_class]}\nTarget={class_names[target_class]}")
        axs[1].axis('off')

      # Perturbation
      axs[2].imshow(perturbation_display.permute(1, 2, 0).numpy())
      axs[2].axis('off')

      plt.tight_layout()
      plt.show()

    # Return the adversarial image for saving
    return adv_img


N_CHANNEL = 3
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, N_CHANNEL)
BATCH_SIZE = 32

def gradient_input(model, x, y_target):
    device = next(model.parameters()).device

    # Get current image dimensions
    h, w, c = x.shape

    # Convert numpy to tensor and add batch dimension
    x_tensor = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    if torch.any(x_tensor) > 1.0:
       x_tensor = x_tensor / 255.0

    # Resize to model input size if needed (model expects IMAGE_SIZE x IMAGE_SIZE)
    if h != IMAGE_SIZE or w != IMAGE_SIZE:
        x_resized = F.interpolate(x_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    else:
        x_resized = x_tensor

    # Set requires_grad on the resized tensor
    x_resized.requires_grad = True

    # Convert one-hot y_target to class index
    target_idx = np.argmax(y_target)
    y_tensor = torch.tensor([target_idx], dtype=torch.long).to(device)

    # Forward pass
    model.eval()
    logits = model(x_resized)

    # Calculate loss
    loss = F.cross_entropy(logits, y_tensor)

    # Compute gradient
    model.zero_grad()
    loss.backward()

    # Get gradient with respect to input
    grad_input = x_resized.grad.data.cpu().numpy()[0]

    # Resize gradient back to original image size if needed
    if h != IMAGE_SIZE or w != IMAGE_SIZE:
        # Convert gradient channels from (C, H, W) to (H, W, C)
        grad_input_hwc = np.transpose(grad_input, (1, 2, 0))
        # Resize using cv2
        grad_input_resized = cv2.resize(grad_input_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
        return grad_input_resized

    # Transpose to match expected shape (height, width, channels)
    return np.transpose(grad_input, (1, 2, 0))


class RandomTransform:
    """
    Random transformation for adversarial attack resilience
    """
    def __init__(self, seed=None, p=1.0, intensity=0.1):
        self.rng = np.random.RandomState(seed)
        self.p = p  # Probability of applying transformation
        self.intensity = intensity
        self.last_transform = None

    def transform(self, img):
        # Apply random transformation with probability p
        if self.rng.random() < self.p:
            # Decide which transformation to apply
            # For simplicity, just implementing a rotation/shift
            angle = self.rng.uniform(-30, 30) * self.intensity
            tx = self.rng.uniform(-10, 10) * self.intensity
            ty = self.rng.uniform(-10, 10) * self.intensity

            # Save transformation parameters
            self.last_transform = (angle, tx, ty)

            # Apply transformation (simplified)
            # In a real implementation, you'd use proper affine transforms
            # Here using a placeholder approach
            from scipy import ndimage
            transformed = ndimage.rotate(img, angle, reshape=False)
            transformed = ndimage.shift(transformed, (ty, tx, 0))
            return np.clip(transformed, 0, 1)

        self.last_transform = (0, 0, 0)  # No transformation
        return img

    def apply_transform(self, img, transform_params):
        angle, tx, ty = transform_params

        # Apply the specified transformation
        from scipy import ndimage
        transformed = ndimage.rotate(img, angle, reshape=False)
        transformed = ndimage.shift(transformed, (ty, tx, 0))
        return np.clip(transformed, 0, 1)

    def get_last_transform(self):
        return self.last_transform


class RandomEnhance:
    """
    Random enhancement for adversarial attack resilience
    """
    def __init__(self, seed=None, p=1.0, intensity=0.1):
        self.rng = np.random.RandomState(seed)
        self.p = p  # Probability of applying enhancement
        self.intensity = intensity
        self.last_factors = None

    def enhance(self, img):
        # Apply random enhancement with probability p
        if self.rng.random() < self.p:
            # Random brightness, contrast, and color adjustments
            brightness = 1.0 + self.rng.uniform(-0.2, 0.2) * self.intensity
            contrast = 1.0 + self.rng.uniform(-0.2, 0.2) * self.intensity
            color = 1.0 + self.rng.uniform(-0.2, 0.2) * self.intensity

            # Save enhancement parameters
            self.last_factors = (brightness, contrast, color)

            # Apply enhancements
            return self.enhance_factors(img, self.last_factors)

        self.last_factors = (1.0, 1.0, 1.0)  # No enhancement
        return img

    def enhance_factors(self, img, factors):
        brightness, contrast, color = factors

        # Apply brightness
        img_bright = img * brightness

        # Apply contrast
        img_mean = np.mean(img_bright, axis=(0, 1), keepdims=True)
        img_contrast = img_mean + (img_bright - img_mean) * contrast

        # Apply color (simplified)
        img_gray = np.mean(img_contrast, axis=2, keepdims=True)
        img_color = img_gray + (img_contrast - img_gray) * color

        return np.clip(img_color, 0, 1)

    def get_last_factors(self):
        return self.last_factors


"""### Define FG"""
def fg(model, x, y, mag_list, target=True, mask=None):
    # Initialize output structure
    x_adv = [[None for _ in range(len(x))] for _ in range(len(mag_list))]
    start_time = time.time()

    for i, x_in in enumerate(x):
        # Retrieve gradient
        if target:
            grad_input_val = -1 * gradient_input(model, x_in, y[i])
        else:
            grad_input_val = gradient_input(model, x_in, y[i])

        # Apply mask if provided
        if mask is not None:
            mask_sample = mask[i]
            mask_rep = np.repeat(mask_sample[:, :, np.newaxis], x_in.shape[2], axis=2)
            grad_input_val *= mask_rep

        # Normalize gradient
        try:
            grad_input_val /= np.linalg.norm(grad_input_val)
        except ZeroDivisionError:
            raise

        # Generate adversarial examples for each magnitude
        for j, mag in enumerate(mag_list):
            x_adv_sample = x_in + grad_input_val * mag
            x_adv_sample = np.clip(x_adv_sample, 0, 1)
            x_adv[j][i] = x_adv_sample

        # Progress printing
        if (i % 1000 == 0) and (i > 0):
            elapsed_time = time.time() - start_time
            print(f"Finished {i} samples in {elapsed_time:.2f}s.")
            start_time = time.time()

    return x_adv


def iterative(model, x, y, norm="2", n_step=20, step_size=0.05, target=True, mask=None):
    x_adv = [None for _ in range(len(x))]
    start_time = time.time()

    for i, x_in in enumerate(x):
        x_cur = np.copy(x_in)

        # Get mask with the same shape as gradient
        if mask is not None:
            mask_rep = np.repeat(mask[i][:, :, np.newaxis], x_in.shape[2], axis=2)

        # Start update in steps
        for _ in range(n_step):
            if target:
                grad_input_val = -1 * gradient_input(model, x_cur, y[i])
            else:
                grad_input_val = gradient_input(model, x_cur, y[i])

            if norm == "2":
                try:
                    grad_input_val /= np.linalg.norm(grad_input_val)
                except ZeroDivisionError:
                    raise
            elif norm == "inf":
                grad_input_val = np.sign(grad_input_val)
            else:
                raise ValueError("Invalid norm!")

            # Apply mask
            if mask is not None:
                grad_input_val *= mask_rep

            x_cur += grad_input_val * step_size
            # Clip to stay in range [0, 1]
            x_cur = np.clip(x_cur, 0, 1)

        x_adv[i] = np.copy(x_cur)

        # Progress printing
        if (i % 200 == 0) and (i > 0):
            elapsed_time = time.time() - start_time
            print(f"Finished {i} samples in {elapsed_time:.2f}s.")
            start_time = time.time()

    return x_adv

def s_pgd(model, x, y, norm="2", n_step=40, step_size=0.01, target=True, mask=None, beta=0.1, early_stop=True):
    # Stochastic Projected Gradient Descent

    # Initialize output list
    x_adv = []
    start_time = time.time()

    for i, x_in in enumerate(x):
        x_cur = np.copy(x_in)
        h, w, c = x_in.shape

        # Get mask with the same shape as gradient
        if mask is not None:
            mask_rep = np.repeat(mask[i][:, :, np.newaxis], c, axis=2)

        # Get target class
        target_class = np.argmax(y[i])

        # Get original prediction
        orig_class, _ = predict(model, x_in)

        # Start update in steps
        for step in range(n_step):
            # Get gradient
            if target:
                grad = -1 * gradient_input(model, x_cur, y[i])
            else:
                grad = gradient_input(model, x_cur, y[i])

            # Get uniformly random direction
            epsilon = np.random.rand(h, w, c) - 0.5

            if norm == "2":
                try:
                    grad_norm = np.linalg.norm(grad)
                    if grad_norm > 0:  # Avoid division by zero
                        grad = grad / grad_norm

                    eps_norm = np.linalg.norm(epsilon)
                    if eps_norm > 0:  # Avoid division by zero
                        epsilon = epsilon / eps_norm
                except:
                    print("Warning: Normalization error in S-PGD")
            elif norm == "inf":
                grad = np.sign(grad)
                epsilon = np.sign(epsilon)
            else:
                raise ValueError("Invalid norm!")

            # Apply mask
            if mask is not None:
                grad *= mask_rep
                epsilon *= mask_rep

            # Update with gradient and random noise
            x_cur += (grad * step_size + beta * epsilon * step_size)

            # Clip to stay in range [0, 1]
            x_cur = np.clip(x_cur, 0, 1)

            if early_stop:
                # Stop when sample becomes adversarial
                current_pred, _ = predict(model, x_cur)

                if target:
                    # For targeted attack, stop when prediction matches target class
                    if current_pred == target_class:
                        print(f"Sample {i} successful at step {step} - reached target class {class_names[target_class]}")
                        break
                else:
                    # For untargeted attack, stop when prediction changes from original
                    if current_pred != orig_class:
                        print(f"Sample {i} successful at step {step} - changed from {class_names[orig_class]} to {class_names[current_pred]}")
                        break

        # Append the adversarial example for this sample
        x_adv.append(x_cur)

        # Progress printing
        if (i % 200 == 0) and (i > 0):
            elapsed_time = time.time() - start_time
            print(f"Finished {i} samples in {elapsed_time:.2f}s.")
            start_time = time.time()

    return x_adv


def jsma(model, x, y, theta=1.0, gamma=0.1, target=True, max_iter=None, clip_min=0.0, clip_max=1.0, mask=None):

    device = next(model.parameters()).device
    n_samples = len(x)
    x_adv = [None for _ in range(n_samples)]

    for i, x_in in enumerate(x):
        print(f"Processing sample {i+1}/{n_samples}")
        h, w, c_channels = x_in.shape

        # If max_iter is not defined, set it based on gamma
        if max_iter is None:
            max_iter = int(gamma * h * w)

        # Create a copy of the input
        x_adv_sample = np.copy(x_in)

        # Get target class index
        target_class = np.argmax(y[i])

        # Get original class
        orig_class, _ = predict(model, x_in)

        # Initialize iteration count
        iteration = 0

        # Create mask for modified pixels (to avoid modifying the same pixel twice)
        modified_pixels = np.zeros((h, w), dtype=bool)

        # Apply user-defined mask if provided
        if mask is not None:
            valid_mask = mask[i].astype(bool)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        # Calculate maximum number of pixels that can be perturbed
        n_pixels = int(gamma * h * w)

        # Main loop
        while iteration < max_iter and np.sum(modified_pixels) < n_pixels:
            # Check if attack already successful
            current_class, _ = predict(model, x_adv_sample)

            if target and current_class == target_class:
                print(f"  Attack successful after {iteration} iterations")
                break
            elif not target and current_class != orig_class:
                print(f"  Attack successful after {iteration} iterations")
                break

            # Calculate Jacobian
            jacobian = compute_jacobian(model, x_adv_sample)

            # Compute saliency map
            saliency_map = compute_saliency_map(
                jacobian, target_class,
                target=target,
                original_class=orig_class,
                valid_mask=(~modified_pixels) & valid_mask
            )

            # If saliency map is empty (no valid pixels left), break
            if np.all(saliency_map == 0):
                print("  No valid pixels left to perturb")
                break

            # Find coordinates of pixel with highest saliency value
            max_idx = np.argmax(saliency_map)
            max_idx = np.unravel_index(max_idx, saliency_map.shape)
            i_pixel, j_pixel = max_idx

            # Apply perturbation to selected pixel (for all channels)
            if target:
                # For targeted attacks, increase pixel values
                x_adv_sample[i_pixel, j_pixel, :] = np.minimum(
                    clip_max, x_adv_sample[i_pixel, j_pixel, :] + theta
                )
            else:
                # For untargeted attacks, decrease pixel values
                x_adv_sample[i_pixel, j_pixel, :] = np.maximum(
                    clip_min, x_adv_sample[i_pixel, j_pixel, :] - theta
                )

            # Mark pixel as modified
            modified_pixels[i_pixel, j_pixel] = True

            # Increment iteration counter
            iteration += 1

            # Print progress every 10 iterations
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}/{max_iter}, pixels modified: {np.sum(modified_pixels)}/{n_pixels}")

        x_adv[i] = x_adv_sample
        print(f"  Sample {i+1} completed: {iteration} iterations, {np.sum(modified_pixels)} pixels modified")

        # Final prediction
        final_class, _ = predict(model, x_adv_sample)
        success = (target and final_class == target_class) or (not target and final_class != orig_class)
        print(f"  Attack {'successful' if success else 'failed'}: Original class: {class_names[orig_class]}, Final class: {class_names[final_class]}")

    return x_adv

def compute_jacobian(model, x_in):
    device = next(model.parameters()).device
    h, w, c = x_in.shape

    # Initialize Jacobian
    with torch.no_grad():
        # Convert to tensor
        x_tensor = torch.tensor(x_in, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        if torch.any(x_tensor) > 1.0:
            x_tensor = x_tensor / 255.0

        # Resize if needed
        if h != IMAGE_SIZE or w != IMAGE_SIZE:
            x_resized = F.interpolate(x_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

        output = model(x_resized)
        num_classes = output.shape[1]

    # Initialize Jacobian
    jacobian = np.zeros((num_classes, h, w))

    # For each class
    for class_idx in range(num_classes):
        # Create a copy of the input that requires gradients
        x_input = torch.tensor(x_in, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        x_input.requires_grad = True

        # Forward pass
        model.eval()

        if torch.any(x_input) > 1.0:
            x_input = x_input / 255.0

        # Resize if needed
        if h != IMAGE_SIZE or w != IMAGE_SIZE:
            x_resized = F.interpolate(x_input, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

        output = model(x_resized)

        # Zero out any existing gradients
        model.zero_grad()

        # Create one-hot vector for the target class
        grad_output = torch.zeros_like(output)
        grad_output[0, class_idx] = 1.0

        # Backward pass
        output.backward(grad_output)

        # Ensure we have gradients
        if x_input.grad is None:
            print(f"Warning: No gradients for class {class_idx}. Using zeros.")
            gradient = np.zeros((c, h, w))
        else:
            # Get gradients
            gradient = x_input.grad.cpu().detach().numpy()[0]

        # Sum over channels to get importance per pixel
        grad_channel_sum = np.sum(gradient, axis=0)
        jacobian[class_idx] = grad_channel_sum

    return jacobian

def compute_saliency_map(jacobian, target_class, target=True, original_class=None, valid_mask=None):
    num_classes, h, w = jacobian.shape

    if target:
        # Targeted attack: maximize target class, minimize others
        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=bool)

        saliency_map = np.zeros((h, w))

        # Calculate direct effect on target class
        target_grad = jacobian[target_class]

        # Calculate sum of effects on other classes
        other_grad_sum = np.sum(jacobian, axis=0) - target_grad

        # Compute saliency map: positive for target class, negative for others
        saliency_map = target_grad * np.abs(other_grad_sum) * (target_grad > 0) * (other_grad_sum < 0)

        # Apply mask
        saliency_map = saliency_map * valid_mask

    else:
        # Untargeted attack: minimize original class
        if original_class is None:
            raise ValueError("Original class must be provided for untargeted attacks")

        if valid_mask is None:
            valid_mask = np.ones((h, w), dtype=bool)

        saliency_map = np.zeros((h, w))

        # Get gradient for original class
        orig_grad = jacobian[original_class]

        # Find most promising other class for each pixel
        max_other_grad = np.zeros((h, w))

        for class_idx in range(num_classes):
            if class_idx != original_class:
                class_grad = jacobian[class_idx]
                # Update max_other_grad where this class has a higher positive gradient
                max_other_grad = np.maximum(max_other_grad, class_grad * (class_grad > 0))

        # Compute saliency map: negative for original class, positive for others
        saliency_map = max_other_grad * np.abs(orig_grad) * (max_other_grad > 0) * (orig_grad < 0)

        # Apply mask
        saliency_map = saliency_map * valid_mask

    return saliency_map


def jsma_clustered(model, x, y, theta=1.0, gamma=0.1, target=True, max_iter=None,
                  clip_min=0.0, clip_max=1.0, mask=None, cluster_size=3, pattern_type='square'):

    device = next(model.parameters()).device
    n_samples = len(x)
    x_adv = [None for _ in range(n_samples)]

    def create_pattern_mask(center_i, center_j, h, w, pattern_type, size):
        """Create a binary mask for the selected pattern centered at (center_i, center_j)"""
        pattern = np.zeros((h, w), dtype=bool)

        for i in range(h):
            for j in range(w):
                # Calculate distance from center
                di = i - center_i
                dj = j - center_j

                # Apply different pattern based on type
                if pattern_type == 'square':
                    # Square pattern
                    if abs(di) <= size and abs(dj) <= size:
                        pattern[i, j] = True

                elif pattern_type == 'circle':
                    # Circle pattern
                    if di*di + dj*dj <= size*size:
                        pattern[i, j] = True

                elif pattern_type == 'cross':
                    # Cross pattern
                    if abs(di) <= size//2 or abs(dj) <= size//2:
                        pattern[i, j] = True

                elif pattern_type == 'random':
                    # Random pattern with higher probability near center
                    dist = np.sqrt(di*di + dj*dj)
                    if dist <= size:
                        # Probability decreases with distance
                        prob = 1.0 - (dist / size) * 0.8
                        if np.random.random() < prob:
                            pattern[i, j] = True

        return pattern

    for i, x_in in enumerate(x):
        print(f"Processing sample {i+1}/{n_samples}")
        h, w, c_channels = x_in.shape

        # If max_iter is not defined, set it based on gamma
        if max_iter is None:
            # Reduce iterations since we're modifying multiple pixels per step
            max_iter = int((gamma * h * w) / (cluster_size * cluster_size))
            max_iter = max(max_iter, 10)  # Ensure at least 10 iterations

        # Create a copy of the input
        x_adv_sample = np.copy(x_in)

        # Get target class index
        target_class = np.argmax(y[i])

        # Get original class
        orig_class, _ = predict(model, x_in)

        # Initialize iteration count
        iteration = 0

        # Create mask for modified pixels (to avoid modifying the same pixel twice)
        modified_pixels = np.zeros((h, w), dtype=bool)

        # Apply user-defined mask if provided
        if mask is not None:
            valid_mask = mask[i].astype(bool)
        else:
            valid_mask = np.ones((h, w), dtype=bool)

        # Calculate maximum number of pixels that can be perturbed
        n_pixels = int(gamma * h * w)

        # Choose color shift based on target class (optional)
        if target:
            # For targeted attacks, we'll use more varied perturbations
            # These values can be adjusted based on the specific target class
            color_shift = np.array([-0.3, -0.3, 0.5])  # Reddish shift for stop signs
        else:
            # For untargeted attacks
            color_shift = np.array([1.0, 1.0, 1.0])  # Uniform shift

        # Main loop
        while iteration < max_iter and np.sum(modified_pixels) < n_pixels:
            # Check if attack already successful
            current_class, probs = predict(model, x_adv_sample)

            if target and current_class == target_class:
                print(f"  Attack successful after {iteration} iterations")
                print(f"  Target class confidence: {probs[target_class]*100:.2f}%")
                break
            elif not target and current_class != orig_class:
                print(f"  Attack successful after {iteration} iterations")
                break

            # Calculate Jacobian
            jacobian = compute_jacobian(model, x_adv_sample)

            # Compute saliency map
            saliency_map = compute_saliency_map(
                jacobian, target_class,
                target=target,
                original_class=orig_class,
                valid_mask=(~modified_pixels) & valid_mask
            )

            # If saliency map is empty (no valid pixels left), break
            if np.all(saliency_map == 0):
                print("  No valid pixels left to perturb")
                break

            # Find coordinates of pixel with highest saliency value
            max_idx = np.argmax(saliency_map)
            max_idx = np.unravel_index(max_idx, saliency_map.shape)
            i_pixel, j_pixel = max_idx

            # Create pattern around the selected pixel
            pattern = create_pattern_mask(i_pixel, j_pixel, h, w, pattern_type, cluster_size)

            # Only apply to valid, unmodified pixels
            pattern = pattern & valid_mask & (~modified_pixels)

            # Apply perturbation to each pixel in the pattern
            pixels_changed = 0

            for pi in range(h):
                for pj in range(w):
                    if pattern[pi, pj]:
                        # Apply weighted perturbation - stronger near center, weaker at edges
                        dist = np.sqrt((pi - i_pixel)**2 + (pj - j_pixel)**2)
                        weight = 1.0
                        if pattern_type != 'square':  # For non-square patterns, apply distance weighting
                            weight = max(0.3, 1.0 - (dist / cluster_size) * 0.7)

                        # Apply perturbation with color shift
                        if target:
                            # For targeted attacks, apply perturbation based on color shift
                            x_adv_sample[pi, pj, :] = np.clip(
                                x_adv_sample[pi, pj, :] + theta * weight * color_shift,
                                clip_min, clip_max
                            )
                        else:
                            # For untargeted attacks, decrease pixel values
                            x_adv_sample[pi, pj, :] = np.clip(
                                x_adv_sample[pi, pj, :] - theta * weight * color_shift,
                                clip_min, clip_max
                            )

                        # Mark pixel as modified
                        modified_pixels[pi, pj] = True
                        pixels_changed += 1

            # If no pixels were changed, break the loop
            if pixels_changed == 0:
                print("  No pixels could be modified in this iteration")
                break

            # Increment iteration counter
            iteration += 1

            # Print progress every 5 iterations
            if iteration % 5 == 0:
                print(f"  Iteration {iteration}/{max_iter}, pixels modified: {np.sum(modified_pixels)}/{n_pixels}")

                # Check current prediction after every 5 iterations
                current_class, current_probs = predict(model, x_adv_sample)
                if target:
                    print(f"  Current prediction: {class_names[current_class]} ({current_probs[current_class]*100:.2f}%)")
                    print(f"  Target confidence: {current_probs[target_class]*100:.2f}%")
                else:
                    print(f"  Current prediction: {class_names[current_class]} ({current_probs[current_class]*100:.2f}%)")
                    print(f"  Original class confidence: {current_probs[orig_class]*100:.2f}%")

        x_adv[i] = x_adv_sample
        print(f"  Sample {i+1} completed: {iteration} iterations, {np.sum(modified_pixels)} pixels modified")

        # Final prediction
        final_class, final_probs = predict(model, x_adv_sample)
        success = (target and final_class == target_class) or (not target and final_class != orig_class)
        print(f"  Attack {'successful' if success else 'failed'}: Original class: {class_names[orig_class]}, Final class: {class_names[final_class]}")

        if target:
            print(f"  Target class confidence: {final_probs[target_class]*100:.2f}%")

    return x_adv

def cw_fast(model, x, y, target=True, c=0.1, kappa=0, lr=0.1, binary_search_steps=3, max_iter=200, mask=None):
    # Carlini and Wagner (C&W) - L2 attack

    device = next(model.parameters()).device
    n_samples = len(x)
    x_adv = [None for _ in range(n_samples)]

    # Use binary search to find optimal c value
    c_lower = np.zeros(n_samples)
    c_upper = np.ones(n_samples) * 10000.0  # Upper bound for c
    c_value = np.ones(n_samples) * c

    best_adv = [np.copy(img) for img in x]  # Initialize with original images
    best_dist = np.ones(n_samples) * np.inf

    # For each binary search step
    for search_step in range(binary_search_steps):
        print(f"Binary search step: {search_step}/{binary_search_steps}")
        start_time = time.time()

        # For each sample
        for i, x_in in enumerate(x):
            h, w, c_channels = x_in.shape

            # Convert to tensor for consistent operations
            x_in_tensor = torch.tensor(x_in, dtype=torch.float32, device=device).unsqueeze(0)

            # Since y is already one-hot encoded, get target class index
            target_class = np.argmax(y[i])

            # Get original class from model prediction
            with torch.no_grad():
                # Convert from NHWC to NCHW
                x_input = x_in_tensor.permute(0, 3, 1, 2)

                if torch.any(x_input) > 1.0:
                    x_input = x_input / 255.0

                # Resize if needed
                if h != IMAGE_SIZE or w != IMAGE_SIZE:
                    x_input = F.interpolate(x_input, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

                logits = model(x_input)
                original_class = logits.argmax(dim=1).item()

            # Prepare mask if provided
            if mask is not None:
                mask_tensor = torch.tensor(
                    np.repeat(mask[i][:, :, np.newaxis], c_channels, axis=2),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)

            # Use the change of variable w where x' = tanh(w)
            # This ensures the adversarial example stays in range [0, 1]
            # First, transform original image to w space
            x_in_scaled = x_in_tensor * 2 - 1  # Scale from [0,1] to [-1,1]
            w = torch.atanh(torch.clamp(x_in_scaled, -0.999, 0.999))
            w_tensor = w.clone().detach().requires_grad_(True)

            # Setup optimizer - using higher learning rate for better progress
            optimizer = torch.optim.Adam([w_tensor], lr=lr)

            # Initialize best attack for this sample
            best_adv_i = x_in.copy()
            best_dist_i = np.inf

            # Early stopping flag
            early_stop = False

            # Optimization loop
            for iteration in range(max_iter):
                optimizer.zero_grad()

                # Convert w to image space using tanh (staying in tensor space)
                adv_images_tanh = torch.tanh(w_tensor)
                adv_images = (adv_images_tanh + 1) / 2  # Scale back to [0,1]

                # Apply mask if provided
                if mask is not None:
                    adv_images = x_in_tensor * (1 - mask_tensor) + adv_images * mask_tensor

                # Convert to NCHW for model
                adv_input = adv_images.permute(0, 3, 1, 2)

                if torch.any(adv_input) > 1.0:
                    adv_input = adv_input / 255.0

                # Resize if needed
                if h != IMAGE_SIZE or w != IMAGE_SIZE:
                    adv_input = F.interpolate(adv_input, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

                # Forward pass
                model.eval()
                logits = model(adv_input)

                # Calculate f(x') based on target/non-target
                if target:
                    # Target case: maximize the target logit
                    target_logit = logits[0, target_class]
                    other_logits = torch.cat([logits[0, :target_class], logits[0, target_class+1:]])
                    max_other_logit = torch.max(other_logits)

                    # f(x') = max(max(Z(x')_i) - Z(x')_t, -kappa)
                    f_value = torch.clamp(max_other_logit - target_logit, min=-kappa)
                else:
                    # Untargeted case: minimize the original logit
                    orig_logit = logits[0, original_class]
                    other_logits = torch.cat([logits[0, :original_class], logits[0, original_class+1:]])
                    max_other_logit = torch.max(other_logits)

                    # f(x') = max(Z(x')_orig - max(Z(x')_i), -kappa)
                    f_value = torch.clamp(orig_logit - max_other_logit, min=-kappa)

                # Calculate L2 distance (in tensor space to maintain gradient)
                delta = adv_images - x_in_tensor
                l2_dist = torch.sqrt(torch.sum(delta**2))

                # CW loss function: c * f(x') + ||x - x'||_2
                loss = c_value[i] * f_value + l2_dist

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Check if this is a better adversarial example
                # First, get current prediction
                with torch.no_grad():
                    curr_pred = model(adv_input).argmax(dim=1).item()

                attack_success = (target and curr_pred == target_class) or (not target and curr_pred != original_class)

                # Convert to numpy for storing and comparison
                adv_images_np = adv_images.detach().cpu().numpy()[0]
                l2_dist_np = l2_dist.item()

                if attack_success and l2_dist_np < best_dist_i:
                    best_dist_i = l2_dist_np
                    best_adv_i = adv_images_np.copy()

                    # Early stopping if we have a good example
                    if l2_dist_np < 2.0 or iteration > max_iter // 2:
                        early_stop = True
                        break

                # Print progress
                if iteration % 50 == 0:
                    print(f"  Sample {i}, Iteration {iteration}, Loss: {loss.item():.4f}, Dist: {l2_dist_np:.4f}")

            # Update best overall results for this sample
            if best_dist_i < best_dist[i]:
                best_dist[i] = best_dist_i
                best_adv[i] = best_adv_i

            # Binary search: update c_value for next search step
            if early_stop or attack_success:
                # Attack succeeded, try lower c
                c_upper[i] = c_value[i]
                if c_upper[i] < 1e9:
                    c_value[i] = (c_lower[i] + c_value[i]) / 2
            else:
                # Attack failed, try higher c
                c_lower[i] = c_value[i]
                if c_upper[i] < 1e9:
                    c_value[i] = (c_value[i] + c_upper[i]) / 2
                else:
                    c_value[i] *= 10

            print(f"Sample {i}/{n_samples}, Best L2: {best_dist[i]:.4f}, c: {c_value[i]:.4f}")

        # After all samples, update x_adv with current best
        x_adv = [np.copy(img) for img in best_adv]

        elapsed = time.time() - start_time
        print(f"Binary search step completed in {elapsed:.2f} seconds")

    return x_adv


def targeted_attack(model, x, target_class_idx, attack_function="fg", **attack_params):
    num_classes = model.fc2.out_features  # Get number of classes from model
    num_samples = len(x)

    # Handle single target class for all images
    if isinstance(target_class_idx, int):
        target_class_idx = [target_class_idx] * num_samples

    # Create one-hot encoded target labels
    target_labels = np.zeros((num_samples, num_classes), dtype=np.float32)
    for i, target_idx in enumerate(target_class_idx):
        target_labels[i, target_idx] = 1.0

    # Choose attack function
    if attack_function == "fg":
        return fg(model, x, target_labels, **attack_params)
    elif attack_function == "iterative":
        return iterative(model, x, target_labels, target=True, **attack_params)
    elif attack_function == "spgd":
        return s_pgd(model, x, target_labels, target=True, **attack_params)
    elif attack_function == "f_cw":
        return cw_fast(model, x, target_labels, target=True, **attack_params)
    elif attack_function == "jsma_clustered":
        return jsma_clustered(model, x, target_labels, target=True, **attack_params)
    elif attack_function == "jsma":
        return jsma(model, x, target_labels, target=True, **attack_params)
    elif attack_function == "deepfool":
        pass
    else:
        raise ValueError(f"Unknown attack function: {attack_function}")


def load_images_from_folder(dataset_path, num_images=10, min_size=100):
    df = pd.read_csv(os.path.join(dataset_path, '_annotations.csv'), header=None)

    images = []
    info = []
    i = 0

    for index, row in df.iterrows():
      filename = row[0]
      ground_truth_label = row[3]
      x1 = row[4]
      y1 = row[5]
      x2 = row[6]
      y2 = row[7]
      w = x2 - x1
      h = y2 - y1

      # Add some Constraint...
      const1 = row[8]
      if "FisheyeCamera" in filename:
        continue
      if const1 == 1 or const1 == 2:
        continue

      filepath = os.path.join(dataset_path, filename)

      if all(dim > min_size for dim in (w, h)) :

        try:
            # Load image using OpenCV
            img = cv2.imread(filepath)
            if img is None:
                raise ValueError(f"Image couldn't be loaded: {filepath}")

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1] range
            img = img / 255.0

            info.append({
                'image': img,
                'filename': filename,
                'filepath': filepath,
                'ground_truth_label': ground_truth_label,
                'w': w,
                'h': h
            })
            i = i + 1

        except Exception as e:
            pass

    return info


"""## **Make dirs**"""
def makedirs():
    # Create directories for saving adversarial examples
    os.makedirs(os.path.join(ADV_SAMPLES_PATH), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'fg'), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'iterative'), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'spgd'), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'jsma'), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'cjsma'), exist_ok=True)
    os.makedirs(os.path.join(ADV_SAMPLES_PATH, 'cw'), exist_ok=True)


"""## **Create Adversarial Dataset for Our Model; using I-FGSM, Stochastic PGD, JSMA, CW (Targetted, Very Aggresive)**"""
def create_adv_samples_batch(model):
    info1 = load_images_from_folder(os.path.join('Dataset','test'),  min_size=50)
    info2 = load_images_from_folder(os.path.join('Dataset','valid'), min_size=50)
    info3 = load_images_from_folder(os.path.join('Dataset','train'), min_size=50)

    all_info = info1 + info2 + info3
    print(all_info)

    all_info.sort(key=lambda item: -1 * item['image'].shape[0])

    for i, info in enumerate(all_info):
        print(f"Image {i} shape: {info['image'].shape}")

    for ii in range(157):
        info = all_info[ii*20:(ii+1)*20]
        
        images = []
        width_sum = 0
        for inf in info:
            images.append(inf['image'])
            width_sum = width_sum + inf['image'].shape[0]
        width_avg = width_sum / 20

        # Set target class
        target_class = 10  # Speed Limit 110

        print("\nPerforming C&W attack...")
        cw_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            attack_function="f_cw",
            c=0.1,
            kappa=-1,
            lr=0.2,
            binary_search_steps=2,
            max_iter=100
        )
        
        print("\nPerforming CJSMA...")
        cjsma_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            attack_function="jsma_clustered",
            theta=2.0, # Strong perturbation
            gamma=0.1, # Allow modifying up to gamma*100% of pixels
            cluster_size=int(width_avg / 10),
            pattern_type='square', # Pattern shape ('square', 'circle', 'cross', 'random')
            max_iter=100
        )

        print("\nPerforming FGSM...")
        fg_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            attack_function="fg",
            mag_list=[2.0]
        )
        fg_examples = fg_examples[0]

        # Iterative attack
        print("\nPerforming Iterative attack...")
        iterative_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            attack_function="iterative",
            norm="2",
            n_step=100,
            step_size=0.2
        )

        # Stochastic PGD attack
        print("\nPerforming Stochastic PGD attack...")
        spgd_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            attack_function="spgd",
            norm="2",
            n_step=75,
            step_size=0.2,
            beta=0.1
        )

        print("\nPerforming JSMA attack...")
        jsma_examples = targeted_attack(
            model,
            images,
            target_class_idx=target_class,
            theta=1.0,
            gamma=0.4,
            attack_function="jsma",
            max_iter=100
        )



        for i, dict_info in enumerate(info):
            orig_filename = dict_info['filename']
            ground_truth_label = class_to_idx[dict_info['ground_truth_label']]

            label_orig, _ = predict(model, images[i])

            if label_orig != ground_truth_label:
                continue

            if label_orig == target_class:
                continue

            label_cjsma, _ = predict(model, cjsma_examples[i])
            label_fg, _ = predict(model, fg_examples[i])
            label_iterative, _ = predict(model, iterative_examples[i])
            label_spgd, _ = predict(model, spgd_examples[i])
            label_jsma, _ = predict(model, jsma_examples[i])
            label_cw , _ = predict(model, cw_examples[i])

            if label_orig != label_cjsma:
                adv_cjsma_filepath = os.path.join(ADV_SAMPLES_PATH, 'cjsma', f"adv_cjsma_{orig_filename}.png")
                adv_img = Image.fromarray((cjsma_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_cjsma_filepath)

            if label_orig != label_fg:
                adv_fg_filepath = os.path.join(ADV_SAMPLES_PATH, 'fg', f"adv_fg_{orig_filename}.png")
                adv_img = Image.fromarray((fg_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_fg_filepath)

            if label_orig != label_iterative:
                adv_iterative_filepath = os.path.join(ADV_SAMPLES_PATH, 'iterative', f"adv_iterative_{orig_filename}.png")
                adv_img = Image.fromarray((iterative_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_iterative_filepath)

            if label_orig != label_spgd:
                adv_spgd_filepath = os.path.join(ADV_SAMPLES_PATH, 'spgd', f"adv_spgd_{orig_filename}.png")
                adv_img = Image.fromarray((spgd_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_spgd_filepath)

            if label_orig != label_jsma:
                adv_jsma_filepath = os.path.join(ADV_SAMPLES_PATH, 'jsma', f"adv_jsma_{orig_filename}.png")
                adv_img = Image.fromarray((jsma_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_jsma_filepath)

            if label_orig != label_cw:
                adv_cw_filepath = os.path.join(ADV_SAMPLES_PATH, 'cw', f"adv_cw_{orig_filename}.png")
                adv_img = Image.fromarray((cw_examples[i] * 255).astype(np.uint8))
                adv_img.save(adv_cw_filepath)



def create_annotation_file(model):
    ANNOTATIONS_CSV_PATH = os.path.join(ADV_SAMPLES_PATH, '_annotations.csv')

    df = {}
    df['train'] = pd.read_csv(os.path.join('Dataset', 'train', '_annotations.csv'), header=None)
    df['test'] = pd.read_csv(os.path.join('Dataset', 'test', '_annotations.csv'), header=None)
    df['valid'] = pd.read_csv(os.path.join('Dataset', 'valid', '_annotations.csv'), header=None)


    with open(ANNOTATIONS_CSV_PATH, "w") as file:
        file.write('ORIGINAL_FILEPATH,ADV_FILEPATH,ATTACK_TYPE,GROUND_TRUTH_LABEL,TARGETTED_CLASS_LABEL,ORIG_PREDICTED_LABEL,ORIG_PREDICTED_PROB,ADV_PREDICTED_LABEL,ADV_PREDICTED_PROB,ADV_TO_BE_RECOVERED_LABEL_PROB\n')

    l1 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'cjsma'))
    l2 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'cw'))
    l3 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'fg'))
    l4 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'iterative'))
    l5 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'jsma'))
    l6 = os.listdir(os.path.join(ADV_SAMPLES_PATH, 'spgd'))

    lists = l1 + l2 + l3 + l4 + l5 + l6

    i = 1
    for adv_filename in lists:
      original_filename = "_".join(adv_filename[4:-4].split('_')[1:])

      original_filepath = None
      dataset = None

      check1 = os.path.isfile(os.path.join('Dataset', 'train', original_filename))
      check2 = os.path.isfile(os.path.join('Dataset', 'test', original_filename))
      check3 = os.path.isfile(os.path.join('Dataset', 'valid', original_filename))

      if check1:
        dataset = 'train'
      elif check2:
        dataset = 'test'
      elif check3:
        dataset = 'valid'
      else:
        print("Not found")
        continue

      original_filepath = os.path.join('Dataset', dataset, original_filename)
      attack_type = adv_filename.split('_')[1]
      adv_filepath = os.path.join(ADV_SAMPLES_PATH, attack_type, adv_filename)

      df_now = df[dataset]

      row = df_now[df_now[0].str.contains(original_filename)]
      ground_truth_label = class_to_idx[row[3].values[0]]
      targeted_class_label = 10

      orig_pred_label, orig_prob = infer(model, cv2.imread(original_filepath))
      orig_pred_prob = orig_prob[orig_pred_label]

      adv_pred_label, adv_prob = infer(model, cv2.imread(adv_filepath))
      adv_pred_prob = adv_prob[adv_pred_label]

      if orig_pred_label == adv_pred_label:
        os.remove(adv_filepath)
        continue
      adv_to_be_recovered_label_prob = adv_prob[orig_pred_label]

      print(f'{i}/{len(lists)}')
      i += 1
      with open(ANNOTATIONS_CSV_PATH, "a") as file:
        file.write(f'{original_filepath},{adv_filepath},{attack_type},{ground_truth_label},{targeted_class_label},{orig_pred_label},{orig_pred_prob},{adv_pred_label},{adv_pred_prob},{adv_to_be_recovered_label_prob}\n')

def main():
    makedirs()
    model = load_model('best_traffic_sign_model.pth')
    #create_adv_samples_batch(model)
    create_annotation_file(model)


if __name__ == '__main__':
    main()