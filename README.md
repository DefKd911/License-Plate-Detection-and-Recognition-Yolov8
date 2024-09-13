# License-Plate-Detection-and-Recognition-Yolov8

![image](https://github.com/user-attachments/assets/b05455f2-4424-43c5-92fa-5742d7448710)

# Overview
This repository contains a project for detecting and recognizing car license plates using YOLOv8 and a Streamlit application. The project involves detecting license plates in images and videos, performing Optical Character Recognition (OCR) to extract text from detected plates, and displaying results with enhanced visual appeal
his project implements a License Plate Detection and Recognition system using YOLOv8 and Streamlit. It allows users to detect and recognize license plates in images and videos using advanced object detection models, and display the results directly in the web app.

# Features
-> Detect License Plates in images and videos

->Recognize and extract the text from license plates

->Display processed results with bounding boxes and confidence scores

-> User-friendly interface using Streamlit

->Supports multiple file formats: .jpg, .jpeg, .png, .mp4, .avi, .mov

# YOLOv8 Model
YOLOv8 (You Only Look Once version 8) is an advanced object detection model designed for real-time detection. 
It improves upon previous versions by enhancing accuracy and speed. YOLOv8 uses a single neural network to predict bounding boxes and class probabilities directly from images, making it suitable for applications requiring fast and accurate object detection.


#  Folder Structure
Car-License-Plate-Recognition-Yolov8/
│
├── yolov8_model/                 # YOLOv8 model files
│   └── kbest.pt
│
├── images/                       # Example images for testing
│   └── example_image.jpg
│
├── videos/                       # Example videos for testing
│   └── example_video.mp4
│
├── app.py                        # Streamlit application script
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
