# Cityguard_vision
### Real-Time Road monitoring system
CityGuard Vision is an AI-powered road safety system designed to detect traffic accidents in real time using CCTV or live video feeds. The system utilizes deep learning and computer vision models to identify accidents and automatically trigger emergency alerts to nearby hospitals and traffic infrastructure.
Beyond accident detection, we are actively expanding CityGuard Vision into a broader urban safety intelligence platform.
We are currently developing additional modules that extend its capabilities to:
- Kidnapping and forced-movement detection : identifying abnormal human behavior patterns such as dragging, sudden forceful movement, or distress-like interactions using temporal motion analysis.
- Suspicious activity monitoring : detecting unusual crowd behavior, aggressive motion clusters, or rapid group convergence that may indicate violence or public safety threats.
- Hit-and-run tracking : tracking vehicles involved in incidents across multiple frames and potentially across multiple cameras.
- Abandoned object detection : identifying unattended bags or suspicious stationary objects in public areas.
- Smart traffic violation detection : detecting wrong-way driving, illegal lane changes, or signal violations using vision-based rule enforcement.
Our long-term goal is to transform CityGuard Vision from a single-purpose accident detection system into a comprehensive AI-driven urban surveillance and response platform that enhances public safety across smart cities.
---

## Project Overview

Road accidents often go unreported for several minutes, delaying emergency response and increasing fatalities. CrashGuard AI solves this problem by continuously monitoring traffic video feeds and automatically detecting accidents using a trained deep learning model.

Once an accident is detected, the system:
- Confirms the event using temporal motion analysis
- Captures and stores accident evidence
- Determines the nearest hospital
- Sends emergency alerts with accident location and image
- Broadcasts accident notifications through VANET-style messaging

This system can be deployed with smart-city CCTV infrastructure to improve road safety and emergency response times.

---

## Key Features

- Real-time accident detection using YOLO-based deep learning models
- Edge AI processing for low latency detection
- Temporal verification to reduce false positives
- Automatic accident image capture and processing
- Nearest hospital detection using location services
- Automated emergency alert messaging
- VANET / V2X style accident broadcast system
- Web interface using Flask for monitoring
- Compatible with CCTV and video feeds

---

## System Architecture

The system consists of multiple modules working together:
### 1. Detection Module
Uses a trained YOLO model to detect potential accident events in video frames.
### 2. Temporal Verification Module
Analyzes frame sequences to verify if the detected event is truly an accident.
### 3. Location Detection
Retrieves the accident location information.
### 4. Nearest Hospital Finder
Uses map APIs to identify the closest hospital to the accident site.
### 5. Alert Messaging System
Sends emergency messages containing accident details.
### 6. Image Upload Module
Uploads captured accident images and converts them into accessible URLs.
### 7. VANET Communication Layer
Broadcasts accident alerts using vehicle network communication concepts.
### 8. Web Interface
Provides a simple Flask web interface to run and monitor the system.

## Project Structure
CrashGuard-AI/
│
├── app.py
├── flask_app.py
├── temporal_module.py
├── currentLocation.py
├── NearestHospital.py
├── Image2Url.py
├── SendMessage.py
├── vanet_layer.py
├── vanet_subscriber.py
│
├── train_accident_model.py
├── train_acci.py
├── model_training.ipynb
│
├── best.pt
├── best_model.pt
├── best_model10.pt
│
├── export_to_onnx.py
├── requirements.txt
└── README.md


# Technologies Used
Python
YOLOv8
OpenCV
Flask
Deep Learning
Computer Vision
VANET Communication
Edge AI
OLA Maps API

# Applications
Smart city traffic monitoring
Automated emergency response systems
Intelligent transportation systems
Highway accident monitoring
Traffic safety analytics

# Future Improvements
Integration with real emergency services
Mobile notification system
Vehicle-to-vehicle alert system
Real-time dashboard for traffic authorities
Improved multi-camera accident tracking
