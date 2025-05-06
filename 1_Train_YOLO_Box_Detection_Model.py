# -*- coding: utf-8 -*-

import locale
import os
import shutil
import torch

from ultralytics import YOLO
from roboflow import Roboflow
from Config import *

rf = Roboflow(api_key=MY_ROBOFLOW_APIKEY)

# FOR TRAINING BOX DETECTION
project_traffic = rf.workspace("selfdriving-car-qtywx").project("self-driving-cars-lfjou")
version = project_traffic.version(6)
dataset = version.download("yolov11")

#BATCH_SIZE = 128  # For A100
#BATCH_SIZE = 80    # FOR T4
BATCH_SIZE = 32    # FOR GTX 1060

def train_box():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  current_directory = os.getcwd()
  absolute_path = os.path.abspath(current_directory)
  dataset_path = os.path.join(absolute_path, 'Self-Driving-Cars-6', 'data.yaml')

  model = YOLO("yolo11n.pt")  # Load a "nano" pretrained model (recommended for training)
  results = model.train(data=dataset_path, epochs=2, batch=BATCH_SIZE, imgsz=640, name='box_detector', single_cls=True, device=device)

train_box()
shutil.rmtree(os.path.join('.', 'Self-Driving-Cars-6'))
