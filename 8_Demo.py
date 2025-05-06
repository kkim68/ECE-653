import cv2
import torch
import numpy as np
import threading
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import time

from Model import class_names, SimpleTrafficSignNet
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import ThreadingLocked

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
model_box = YOLO("best.pt")
model_traffic = SimpleTrafficSignNet(num_classes=13)
model_traffic.load_state_dict(torch.load('best_traffic_sign_model.pth', map_location=device))
model_traffic.eval()

# Global variables
cur_box = (-1, -1, -1, -1)
class_name = None
class_conf = None
apply_filter = False

# For storing current class probabilities for plotting
class_probabilities = {}
last_update_time = time.time()
update_interval = 0.2  # Update graph every 0.2 seconds

lock = threading.Lock()


if not hasattr(SimpleTrafficSignNet, 'predict_all_probs'):
    def predict_all_probs(self, img):
        """
        Get probabilities for all classes
        This is a mock implementation - modify to match your model's architecture
        """
        import torch.nn.functional as F
        
        # Convert PIL image to tensor
        if isinstance(img, Image.Image):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            img_tensor = transform(img).unsqueeze(0).to(device)
        else:
            # Assume it's already a tensor
            img_tensor = img
            
        # Get logits from the model
        with torch.no_grad():
            logits = self(img_tensor)
            probabilities = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Create dictionary of class names and probabilities
        # Modify these class names to match your actual classes
        
        result = {}
        for i, prob in enumerate(probabilities):
            if i < len(class_names):
                result[class_names[i]] = float(prob)
            else:
                result[f"Class {i}"] = float(prob)
                
        return result
        
    # Add method to the class
    SimpleTrafficSignNet.predict_all_probs = predict_all_probs

@ThreadingLocked()
def predict(frame, img):
    global cur_box, class_name, class_conf, class_probabilities

    detections = model_box(frame, verbose=False)

    detected = False
    for detection in detections:
        for i, bbox in enumerate(detection.boxes):
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            detected = True
            cropped = img.crop((x1, y1, x2, y2))
            
            # Get all class probabilities
            all_probs = model_traffic.predict_all_probs(cropped)

            cc = -1
            cn = ''

            for i, class_name in enumerate(all_probs):
                if cc < all_probs[class_name]:
                    cc = all_probs[class_name]
                    cn = class_name

            lock.acquire()
            try:
                cur_box = (x1, y1, x2, y2)
                class_name = cn
                class_conf = cc
                
                # Update class probabilities for the graph
                class_probabilities = all_probs
            finally:
                lock.release()

    if not detected:
        lock.acquire()
        try:
            cur_box = (-1, -1, -1, -1)
            class_name = None
            class_conf = None
            # Clear probabilities when no detection
            class_probabilities = {}
        finally:
            lock.release()

def click_and_crop(event, x, y, flags, param):
    global apply_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        if apply_filter:
            apply_filter = False
        else:
            apply_filter = True

def create_seaborn_plot():
    # Create a bar plot of current class probabilities
    fig = Figure(figsize=(8, 4), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Create a DataFrame for Seaborn from our class probabilities
    if class_probabilities:        
        #sorted_items = sorted(class_probabilities.items(), key=lambda x: x[1], reverse=True)
        sorted_items = class_probabilities.items()
        class_names = [item[0] for item in sorted_items]
        probs = [item[1] for item in sorted_items]
        
        df = pd.DataFrame({
            'Class': class_names,
            'Probability': probs
        })
        
        # Create the Seaborn bar plot
        sns.set_style("darkgrid")
        bars = sns.barplot(x='Class', y='Probability', data=df, ax=ax, palette='viridis')
        
        # Highlight the class with highest probability
        if class_names:
            max_prob_index = probs.index(max(probs))
            bars.patches[max_prob_index].set_facecolor('red')
        
        # Add value labels on top of bars
        for i, p in enumerate(bars.patches):
            bars.annotate(f'{probs[i]:.2f}', 
                         (p.get_x() + p.get_width() / 2., p.get_height()), 
                         ha = 'center', va = 'bottom',
                         fontsize=8,
                         xytext = (0, 3), 
                         textcoords = 'offset points')
                         
        ax.set_title('Traffic Sign Class Probabilities')
        ax.set_ylim(0, 1.1)  # Leave room for annotations
        ax.tick_params(axis='x', rotation=45)  # Rotate x labels for better readability
        fig.tight_layout()
        
        # Convert plot to an OpenCV image
        canvas.draw()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        return img
    
    return None

def main():
    global last_update_time
    
    print(f"Using device: {device}")

    filter1 = np.load('best_filter_1.npy')
    filter2 = np.load('best_filter_2.npy')    

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    cap.set(cv2.CAP_PROP_FPS, 60)  # This does not work for all cameras...
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Set height

    # Create a window to display the graph
    cv2.namedWindow('Webcam')
    cv2.namedWindow('Class Probabilities', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Class Probabilities', 500, 250)
    
    # Initialize the graph
    graph_img = np.zeros((250, 500, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(graph_img, "Waiting for detection...", (70, 120), font, 1, (255, 255, 255), 2)
    cv2.imshow('Class Probabilities', graph_img)
    
    # Variable to keep track of the last valid graph
    last_valid_graph = graph_img
    
    count = 0
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2image = frame
        h = cv2image.shape[0]
        
        if apply_filter:
            original_mean = np.mean(cv2image, axis=(0, 1), keepdims=True)
            filtered_image = cv2.filter2D(cv2image, -1, filter1, borderType=cv2.BORDER_REFLECT)
            filtered_image = cv2.filter2D(filtered_image, -1, filter2, borderType=cv2.BORDER_REFLECT)
            filtered_mean = np.mean(filtered_image, axis=(0, 1), keepdims=True)
            brightness_ratio = original_mean / (filtered_mean + 1e-10)
            filtered_image = filtered_image * brightness_ratio
            filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
            cv2image = filtered_image

        # Create a PIL image
        img = Image.fromarray(cv2image)

        # Run prediction
        predict(Image.fromarray(frame), img)

        # Draw bounding box and label
        if min(cur_box) != -1:
            text = f'{class_name} ({class_conf*100:.2f}%)'
            org = (cur_box[0], cur_box[1])  # Bottom-left corner of the text string
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            color = (0, 255, 0)  # Green color (BGR)
            thickness = 2
            cv2.putText(cv2image, text, org, font, font_scale, color, thickness)
            cv2.rectangle(cv2image, (cur_box[0], cur_box[1] + 10), (cur_box[2], cur_box[3]), (255, 0, 0), 3)

        text = 'Filter OFF'
        if apply_filter:
            text = 'Filter ON'
            
        cv2.putText(cv2image, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # Update the Seaborn graph at regular intervals
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            if class_probabilities:  # If we have class probabilities data
                new_graph = create_seaborn_plot()
                if new_graph is not None:
                    graph_img = new_graph
                    last_valid_graph = graph_img  # Save the last valid graph
            else:
                # If no new detection but we have a previous graph, keep showing it
                if 'last_valid_graph' in locals() and last_valid_graph is not None:
                    graph_img = last_valid_graph
            
            last_update_time = current_time
        
        # Display the webcam feed and graph
        cv2.imshow('Webcam', cv2image)
        cv2.imshow('Class Probabilities', graph_img)
            
        cv2.setMouseCallback("Webcam", click_and_crop)

        count = count + 1
        if count > 5000:
            count = 0

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()