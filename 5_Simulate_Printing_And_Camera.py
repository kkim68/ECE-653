import pandas as pd
import cv2
import os
import numpy as np
import random
import shutil

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


def modify_column(series):
  return series.replace('adv_samples', 'adv_samples_printed_and_pictured')

def simulate_print_and_camera():
    pd_dataset = pd.read_csv('adv_samples/_annotations.csv')

    base_dir = 'adv_samples_printed_and_pictured'
    os.makedirs('adv_samples_printed_and_pictured', exist_ok=True)
    os.makedirs('adv_samples_printed_and_pictured/cjsma', exist_ok=True)
    os.makedirs('adv_samples_printed_and_pictured/iterative', exist_ok=True)
    os.makedirs('adv_samples_printed_and_pictured/spgd', exist_ok=True)

    mask = pd_dataset['ATTACK_TYPE'].str.match('jsma')
    selected_samples = pd_dataset[~mask].sort_values(by=['ORIGINAL_FILEPATH'])

    mask = selected_samples['ATTACK_TYPE'].str.contains('cw')
    selected_samples = selected_samples[~mask].sort_values(by=['ORIGINAL_FILEPATH'])

    print(selected_samples['ADV_FILEPATH'])
    for index, row in selected_samples.iterrows():
        adv_filepath = row['ADV_FILEPATH']
        attack_type = row['ATTACK_TYPE']
        img = cv2.imread(adv_filepath)
        printed = simulate_print(img, dpi=144, paper_type='matte')
        pictured = simulate_camera(printed, 0.316, 0, 1.0, 0, 1)
        dst_filepath = row['ADV_FILEPATH'].replace('adv_samples', 'adv_samples_printed_and_pictured')
        cv2.imwrite(dst_filepath, pictured, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    selected_samples['ADV_FILEPATH'] = selected_samples['ADV_FILEPATH'].apply(modify_column)
    selected_samples.to_csv(os.path.join(base_dir, '_annotations.csv'), index=False)


def make_dataset_for_experiment():
    df = pd.read_csv(os.path.join('adv_samples', '_annotations.csv'))
    orig_files = df['ORIGINAL_FILEPATH'].unique()

    sample_path = 'adv_samples_for_experiment'
    os.makedirs(sample_path, exist_ok=True)

    idx = 1

    for orig_file in orig_files:
        filtered = df[df['ORIGINAL_FILEPATH'] == orig_file]
        
        count = 0
        count = count + len(filtered[filtered['ATTACK_TYPE'] == 'spgd']) 
        count = count + len(filtered[filtered['ATTACK_TYPE'] == 'cjsma']) 
        count = count + len(filtered[filtered['ATTACK_TYPE'] == 'jsma']) 
        count = count + len(filtered[filtered['ATTACK_TYPE'] == 'cw']) 
        count = count + len(filtered[filtered['ATTACK_TYPE'] == 'iterative'])

        if count == 5:
            base_folder = os.path.join(sample_path, str(idx))
            os.makedirs(base_folder, exist_ok=True)
            shutil.copy(orig_file, base_folder)

            for index, row in filtered.iterrows():
                src = row['ADV_FILEPATH']
                shutil.copy(src, base_folder)

            idx = idx + 1

def main():
    simulate_print_and_camera()
    make_dataset_for_experiment()

main()
