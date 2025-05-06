# -*- coding: utf-8 -*-
import shutil
import locale
import cv2
import os

from roboflow import Roboflow
from Config import *

locale.getpreferredencoding = "UTF-8"
rf = Roboflow(api_key=MY_ROBOFLOW_APIKEY)

# BELOW ARE ALL FOR TRAINING CLASSIFICATION
project_traffic = rf.workspace("selfdriving-car-qtywx").project("self-driving-cars-lfjou")
version = project_traffic.version(6)
dataset = version.download("tensorflow")

project = rf.workspace("school-0ljld").project("kaggle-datasets-for-traffic")
version = project.version(2)
dataset = version.download("tensorflow")

project = rf.workspace("trafic-object-detection-autonomous-no-augm").project("traffic-sign-detection-0y9yn")
version = project.version(2)
dataset = version.download("tensorflow")

MIN_SIZE = 64 # filtering only width and height greater than 64

roboflow_data_paths = ["./Self-Driving-Cars-6",
                       "./kaggle-datasets-for-traffic-2",
                       "./Traffic-sign-detection-2"]

our_dataset_path = "./Dataset"
our_dataset_test_path = os.path.join(our_dataset_path, 'test')
our_dataset_train_path = os.path.join(our_dataset_path, 'train')
our_dataset_valid_path = os.path.join(our_dataset_path, 'valid')

os.makedirs(our_dataset_path, exist_ok=True)
os.makedirs(our_dataset_test_path, exist_ok=True)
os.makedirs(our_dataset_train_path, exist_ok=True)
os.makedirs(our_dataset_valid_path, exist_ok=True)

folders = ['test', 'train', 'valid']
target_labels = [ 'Speed Limit 10',
                  'Speed Limit 20',
                  'Speed Limit 30',
                  'Speed Limit 40',
                  'Speed Limit 50',
                  'Speed Limit 60',
                  'Speed Limit 70',
                  'Speed Limit 80',
                  'Speed Limit 90',
                  'Speed Limit 100',
                  'Speed Limit 110',
                  'Speed Limit 120',
                  'Stop']


for index, roboflow_data_path in enumerate(roboflow_data_paths):
  for data_type in folders:
    with open(os.path.join(roboflow_data_path, data_type, '_annotations.csv'), 'r') as fh:
      lines = fh.readlines()

      for line in lines:
        if line.strip() != '' and not line.startswith('filename'):
          splitted = line.strip().split(',')

          filename = splitted[0]
          label = splitted[3]
          if label not in target_labels:
            continue

          x1 = splitted[4]
          y1 = splitted[5]
          x2 = splitted[6]
          y2 = splitted[7]
          if int(x2) - int(x1) < MIN_SIZE or int(y2) - int(y1) < MIN_SIZE:
            continue

          # copy image
          img = cv2.imread(os.path.join(roboflow_data_path, data_type, filename))
          img = img[int(y1):int(y2), int(x1):int(x2)]
          cv2.imwrite(os.path.join(our_dataset_path, data_type, filename), img)

          with open(os.path.join(our_dataset_path, data_type,'_annotations.csv'), "a") as file:
            file.write(line.strip() + ',' + str(index) + '\n')

    with open(os.path.join(our_dataset_path, data_type,'_annotations.csv'), "a") as file:
      file.write("\n")

for path in roboflow_data_paths:
  shutil.rmtree(path)