import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm

import os
import time
import random

from Model import *
from Config import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model('best_traffic_sign_model.pth')
ADV_SAMPLES_PATH = os.path.join('adv_samples_for_experiment')


# Source code for printing simulation is created by Claude.ai
# prompt: "generate python code that simulates paper print ..."

# We can't print 3,000 pages of adversarial samples and take photo of them 
# to train our filters...
def simulate_print(img, dpi=300, paper_type='glossy'):
    """
    Simulate how an image would look when printed on paper.

    Parameters:
    - img: cv2 image
    - output_path: Path to save the simulated output
    - dpi: Dots per inch for the simulated print (default 300)
    - paper_type: Type of paper ('glossy', 'matte', 'newspaper')

    Returns:
    - Simulated printed image
    """

    # Step 1: Resize based on DPI
    # Calculate scaling factor based on DPI
    # For example, 144 DPI (screen) to print DPI (e.g., 300)
    scale_factor = 144 / dpi
    if scale_factor != 1:
        new_width = int(img.shape[1] * scale_factor)
        new_height = int(img.shape[0] * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # Resize back to original dimensions to simulate the print resolution
        img = cv2.resize(img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Step 2: Apply color profile adjustments (CMYK simulation)
    # Convert to float32 for processing
    img_float = img.astype(np.float32) / 255.0

    # Apply color adjustments based on paper type
    if paper_type == 'glossy':
        # Glossy paper - vibrant but slightly darkened
        contrast = 1.1
        brightness = -0.05
        saturation = 1.05
    elif paper_type == 'matte':
        # Matte paper - less contrast, slightly desaturated
        contrast = 0.90
        brightness = 0.0
        saturation = 0.9
    elif paper_type == 'newspaper':
        # Newspaper - low contrast, desaturated
        contrast = 0.8
        brightness = 0.1
        saturation = 0.7
    else:
        # Default
        contrast = 1.0
        brightness = 0.0
        saturation = 1.0

    # Apply contrast and brightness
    img_float = img_float * contrast + brightness
    img_float = np.clip(img_float, 0, 1)

    # Apply saturation adjustment
    hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = hsv[:,:,1] * saturation
    img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 3: Simulate ink spread/dot gain
    # Create a slight blur to simulate ink spreading on paper
    #kernel_size = max(1, int(300 / dpi))  # Adjust based on DPI
    #img_float = cv2.GaussianBlur(img_float, (kernel_size*2+1, kernel_size*2+1), 0.5)
    kernel_size = 2
    #img_float = cv2.GaussianBlur(img_float, (kernel_size*2+1, kernel_size*2+1), 0.5)
    img_float = cv2.GaussianBlur(img_float, (kernel_size*2+1, kernel_size*2+1), 0.5)
    img_float = cv2.medianBlur(img_float, 1)


    # Step 4: Add paper texture
    # Create a paper texture
    texture = np.ones_like(img_float)
    noise = np.random.normal(loc=0.95, scale=0.05, size=img_float.shape)
    texture = texture * noise
    texture = np.clip(texture, 0, 1)

    # Blend the image with the paper texture
    if paper_type == 'glossy':
        blend_factor = 0.03
    elif paper_type == 'newspaper':
        blend_factor = 0.15
    else:  # matte
        blend_factor = 0.08

    img_float = img_float * (1 - blend_factor) + texture * blend_factor

    # Convert back to uint8
    img_result = (img_float * 255).astype(np.uint8)

    return img_result

"""## **Simulate Image Resolution**"""

def simulate_camera(img,
                    target_resolution_ratio=0.5,
                    sensor_noise=0.02,
                    lens_quality=0.9,
                    sharpening=1.0,
                    brightness_preserve=1.0):  # Added brightness preservation parameter
    """
    Simulate taking a photo with a camera at different resolutions.

    Parameters:
    - image_path: cv2 image of high-res version
    - output_path: Path to save the simulated output (if None, just returns the image)
    - target_resolution: Tuple of (width, height) for the target camera resolution
    - sensor_noise: Amount of sensor noise to add (0.0 to 1.0)
    - lens_quality: Simulates lens quality (0.5 to 1.0, higher is better)
    - sharpening: Amount of camera sharpening to apply (0.0 to 2.0)
    - brightness_preserve: Factor to preserve original brightness (0.8 to 1.2)

    Returns:
    - Simulated camera image
    """

    original_height, original_width = img.shape[:2]
    target_resolution = (int(original_width * target_resolution_ratio), int(original_height * target_resolution_ratio))

    # Step 0: Random Rotation

    # Generate random angle between -3 and 3 degrees
    angle = random.uniform(-3, 3)

    # Get rotation matrix
    center = (original_width // 2, original_height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calculate new bounding dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(original_height * abs_sin + original_width * abs_cos)
    new_height = int(original_height * abs_cos + original_width * abs_sin)

    # Adjust the rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]

    # Perform the rotation with white background (255,255,255)
    rotated = cv2.warpAffine(img, rotation_matrix, (new_width, new_height),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(255, 255, 255))

    # Crop to original size from the center to avoid white borders
    start_x = (new_width - original_width) // 2
    start_y = (new_height - original_height) // 2
    rotated = rotated[start_y:start_y+original_height, start_x:start_x+original_width]

    # If cropping caused any dimension issues, resize back to original
    if rotated.shape[:2] != (original_height, original_width):
        rotated = cv2.resize(rotated, (original_width, original_height))

    img = rotated

    # Step 1: Downsample to target resolution (simulating the camera sensor)
    downsampled = cv2.resize(img, target_resolution, interpolation=cv2.INTER_AREA)

    # Step 2: Apply lens blur based on lens quality
    # Lower quality lenses have more blur
    blur_amount = int(max(1, (1.0 - lens_quality) * 10))
    if blur_amount > 1:
        downsampled = cv2.GaussianBlur(downsampled, (blur_amount*2+1, blur_amount*2+1), 0)

    # Step 3: Add sensor noise
    if sensor_noise > 0:
        # Convert image to float32 for the addition operation
        float_img = downsampled.astype(np.float32)
        # Create noise with the same data type as the float image
        noise = np.random.normal(0, sensor_noise * 255, float_img.shape).astype(np.float32)
        # Add noise
        noisy_img = cv2.add(float_img, noise)
        # Convert back to uint8 with clipping
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    else:
        noisy_img = downsampled

    # Step 4: Apply camera sharpening (most cameras do this in-camera)
    if sharpening > 0:
        # Create a sharpening kernel
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + sharpening, -1],
                           [-1, -1, -1]]) / (sharpening + 5)
        sharpened = cv2.filter2D(noisy_img, -1, kernel)
    else:
        sharpened = noisy_img

    # Step 5: Simulate color processing/enhancement that cameras do
    # With brightness preservation
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Boost saturation slightly while preserving brightness
    hsv[:,:,1] = hsv[:,:,1] * 1.1  # Boost saturation slightly
    hsv[:,:,2] = hsv[:,:,2] * brightness_preserve  # Preserve brightness with adjustable factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    color_processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Step 6: JPEG compression artifacts (optional)
    # This simulates the camera's JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded_img = cv2.imencode('.jpg', color_processed, encode_param)
    simulated_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)

    # Step 7: Resize back to original resolution for comparison
    # This simulates viewing the photo on a display
    result = cv2.resize(simulated_img, (original_width, original_height),
                        interpolation=cv2.INTER_LINEAR)

    return result

"""## **Apply Filter**"""

filter1 = np.load('best_filter_1.npy')
filter2 = np.load('best_filter_2.npy')

def apply_our_filter(img):
  original_mean = np.mean(sample_cv2_image, axis=(0, 1), keepdims=True)
  filtered_image = cv2.filter2D(sample_cv2_image, -1, filter1, borderType=cv2.BORDER_REFLECT)
  filtered_image = cv2.filter2D(filtered_image, -1, filter2, borderType=cv2.BORDER_REFLECT)
  filtered_mean = np.mean(filtered_image, axis=(0, 1), keepdims=True)
  brightness_ratio = original_mean / (filtered_mean + 1e-10)
  filtered_image = filtered_image * brightness_ratio
  filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
  return filtered_image

"""## **Show Result**"""

plt.rcParams.update({'font.size': 15})

samples = [476, 304, 162, 114, 375, 183, 206, 464, 387, 39, 226, 465, 263, 344, 378, 473, 79, 326, 191, 73, 93, 437, 131, 439, 161, 12, 284, 144, 125, 121, 26, 309, 219, 35, 9, 19, 410, 409, 314, 209, 459, 176, 167, 468, 450, 216, 324, 360, 392]
samples_to_show = [476, 304, 162, 114, 375, 162, 114, 304, 183, 206, 464, 387]

printed_recovered_orig_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}

printed_reduced_adv_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}

pictured_recovered_orig_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}

pictured_reduced_adv_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}


filtered_recovered_orig_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}

filtered_reduced_adv_label_pred_sum = {
    'cjsma': 0,
    'jsma': 0,
    'cw': 0,
    'iterative': 0,
    'spgd': 0
}

for i in samples:
  fig = None
  axes = None

  if i in samples_to_show:
    fig, axes = plt.subplots(4, 6, figsize=(30, 30), dpi=36)
    for l1 in range(4):
      for l2 in range(6):
        axes[l1][l2].axis('off')

  basepath = os.path.join(ADV_SAMPLES_PATH, f'{i}')
  files = os.listdir(basepath)
  orig = list(filter(lambda string: "adv_" not in string, files))
  fg = list(filter(lambda string: "_fg_" in string, files))
  files.remove(orig[0])
  if len(fg):
    files.remove(fg[0])

  file_path = os.path.join(basepath, orig[0])
  sample_cv2_image = cv2.imread(file_path)
  sample_cv2_image = cv2.cvtColor(sample_cv2_image, cv2.COLOR_BGR2RGB)
  orig_class_idx, pred_orig = infer(model, sample_cv2_image)

  orig_printed = simulate_print(sample_cv2_image, dpi=144, paper_type='matte')
  orig_printed_class_idx, pred_printed = infer(model, orig_printed)

  orig_pictured = simulate_camera(orig_printed, 0.316, 0, 1.0, 0, 1)
  orig_pictured_class_idx, pred_pictured = infer(model, orig_pictured)

  orig_filtred = apply_our_filter(orig_pictured)
  orig_filtred_class_idx, pred_filtred = infer(model, orig_filtred)

  if i in samples_to_show:
    axes[0][0].imshow(sample_cv2_image)
    axes[0][0].set_title(f'Original (Fully Digital)\n{class_names[orig_class_idx]} ({pred_orig[orig_class_idx]*100:.1f}%)\n-')
    axes[1][0].imshow(orig_printed)
    axes[1][0].set_title(f'Original (Printed+50MP)\n{class_names[orig_printed_class_idx]} ({pred_printed[orig_printed_class_idx]*100:.1f}%)\n-')
    axes[2][0].imshow(orig_pictured)
    axes[2][0].set_title(f'Original (Downsampled to 5MP)\n{class_names[orig_pictured_class_idx]} ({pred_pictured[orig_pictured_class_idx]*100:.1f}%)\n-')
    axes[3][0].imshow(orig_filtred)
    axes[3][0].set_title(f'Original (Filtered)\n{class_names[orig_filtred_class_idx]} ({pred_filtred[orig_filtred_class_idx]*100:.1f}%)\n-')

  for j, filename in enumerate(sorted(files)):
    # print(filename)
    file_path = os.path.join(basepath, filename)
    sample_cv2_image = cv2.imread(file_path)
    sample_cv2_image = cv2.cvtColor(sample_cv2_image, cv2.COLOR_BGR2RGB)
    adv_class_idx, pred_adv = infer(model, sample_cv2_image)

    filename_split = filename.split('_')
    attack_type = filename_split[1]

    printed = simulate_print(sample_cv2_image, dpi=144, paper_type='matte')
    printed_class_idx, pred_adv_printed = infer(model, printed)

    # Where did 0.316 came from?
    # sqrt(5) / sqrt(50)
    pictured = simulate_camera(printed, 0.316, 0, 1.0, 0, 1)
    pictured_class_idx, pred_adv_pictured = infer(model, pictured)

    filtered = apply_our_filter(pictured)
    filtered_class_idx, pred_adv_filtered = infer(model, filtered)

    printed_recovered_orig_label_pred_sum[attack_type] += (pred_adv_printed[orig_class_idx] - pred_adv[orig_class_idx])
    printed_reduced_adv_label_pred_sum[attack_type] += (pred_adv_printed[adv_class_idx] - pred_adv[adv_class_idx])

    pictured_recovered_orig_label_pred_sum[attack_type] += (pred_adv_pictured[orig_class_idx] - pred_adv[orig_class_idx])
    pictured_reduced_adv_label_pred_sum[attack_type] += (pred_adv_pictured[adv_class_idx] - pred_adv[adv_class_idx])

    filtered_recovered_orig_label_pred_sum[attack_type] += (pred_adv_filtered[orig_class_idx] - pred_adv[orig_class_idx])
    filtered_reduced_adv_label_pred_sum[attack_type] += (pred_adv_filtered[adv_class_idx] - pred_adv[adv_class_idx])

    if i in samples_to_show:
      axes[0][j+1].imshow(sample_cv2_image)
      axes[0][j+1].set_title(f'{attack_type} (Fully Digital)\n{class_names[orig_class_idx]} ({pred_adv[orig_class_idx]*100:.1f}%)\n{class_names[adv_class_idx]} ({pred_adv[adv_class_idx]*100:.1f}%)')
      axes[1][j+1].imshow(printed)
      axes[1][j+1].set_title(f'{attack_type} (Printed+50MP)\n{class_names[orig_class_idx]} ({pred_adv_printed[orig_class_idx]*100:.1f}%)\n{class_names[adv_class_idx]} ({pred_adv_printed[adv_class_idx]*100:.1f}%)')
      axes[2][j+1].imshow(pictured)
      axes[2][j+1].set_title(f'{attack_type} (Downsampled to 5MP)\n{class_names[orig_class_idx]} ({pred_adv_pictured[orig_class_idx]*100:.1f}%)\n{class_names[adv_class_idx]} ({pred_adv_pictured[adv_class_idx]*100:.1f}%)')
      axes[3][j+1].imshow(filtered)
      axes[3][j+1].set_title(f'{attack_type} (Filtered)\n{class_names[orig_class_idx]} ({pred_adv_filtered[orig_class_idx]*100:.1f}%)\n{class_names[adv_class_idx]} ({pred_adv_filtered[adv_class_idx]*100:.1f}%)')

  if i in samples_to_show:
    fig.tight_layout()
    plt.tight_layout()
    plt.show()


attack_types = ['cjsma', 'jsma', 'cw', 'iterative', 'spgd']

print (f'[PRINTED]')
for attack_type in attack_types:
  avg_recovery = printed_recovered_orig_label_pred_sum[attack_type] / len(samples)
  avg_pert_destroy = printed_reduced_adv_label_pred_sum[attack_type] / len(samples)
  print(f'Attack Type: {attack_type}, Avg Original Label Receovery: {avg_recovery*100:.2f}%, Avg Perturbation Loss: {avg_pert_destroy*100:.2f}%')

print (f'[PICTURED]')
for attack_type in attack_types:
  avg_recovery = pictured_recovered_orig_label_pred_sum[attack_type] / len(samples)
  avg_pert_destroy = pictured_reduced_adv_label_pred_sum[attack_type] / len(samples)
  print(f'Attack Type: {attack_type}, Avg Original Label Receovery: {avg_recovery*100:.2f}%, Avg Perturbation Loss: {avg_pert_destroy*100:.2f}%')

print (f'[FILTERED]')
for attack_type in attack_types:
  avg_recovery = filtered_recovered_orig_label_pred_sum[attack_type] / len(samples)
  avg_pert_destroy = filtered_reduced_adv_label_pred_sum[attack_type] / len(samples)
  print(f'Attack Type: {attack_type}, Avg Original Label Receovery: {avg_recovery*100:.2f}%, Avg Perturbation Loss: {avg_pert_destroy*100:.2f}%')