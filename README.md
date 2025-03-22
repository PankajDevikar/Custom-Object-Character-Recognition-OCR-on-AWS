# Custom Object Character Recognition (OCR) on AWS Using Appwrite Integration

**Building a Custom OCR System by Combining YOLOv3, Tesseract, and Appwrite**


## 1. Overview

This project implements a custom object character recognition (OCR) system designed to extract key information from lab reports. The system integrates a YOLOv3-based text detection model with the Tesseract OCR engine, and it uses Appwrite for cloud storage and API management. The solution automates the process of converting lab reports into editable text, and it is deployed on AWS for scalability.

---

## 2. Features

- **YOLOv3 Text Detection:**  
  Detects regions of interest (text regions) within lab report images.

- **Tesseract OCR Integration:**  
  Extracts text from the detected regions.

- **Appwrite Integration:**  
  Utilizes Appwrite for cloud-based data management and API endpoints.

- **AWS Deployment:**  
  Designed to run on AWS for scalable and cost-efficient processing.

- **User Interface:**  
  A Streamlit app provides an interactive interface for uploading images, viewing detections, and displaying OCR results.

---

## 3. Architecture

### 3.1 Overall System Diagram

![image](https://github.com/PankajDevikar/Custom-Object-Character-Recognition-OCR-on-AWS/blob/main/Screenshorts/img.png)


### 3.2 Detailed Architecture

The system consists of three main components:
- **Detection Module:**  
  Uses a YOLOv3-based model to detect text regions.
- **OCR Module:**  
  Uses Tesseract OCR to extract text from detected regions.
- **Cloud & API Module:**  
  Appwrite manages storage and API endpoints for scalable data handling.

### 3.3 Appwrite Integration

Appwrite is used to store training data, model outputs, and to provide RESTful API endpoints for accessing results. This enables real-time updates and monitoring through a user-friendly dashboard.

---

## 4. Installation

### Prerequisites

- Python 3.x  
- Virtual environment (recommended)  
- AWS Account and proper credentials  
- Appwrite instance for backend management


## 📸 Output Screenshots

Below are the screenshots demonstrating the project output:

![image](https://github.com/PankajDevikar/Custom-Object-Character-Recognition-OCR-on-AWS/blob/main/Screenshorts/img2.png)
![image](https://github.com/PankajDevikar/Custom-Object-Character-Recognition-OCR-on-AWS/blob/main/Screenshorts/img3.png)
![image](https://github.com/PankajDevikar/Custom-Object-Character-Recognition-OCR-on-AWS/blob/main/Screenshorts/img4.png)
![image](https://github.com/PankajDevikar/Custom-Object-Character-Recognition-OCR-on-AWS/blob/main/Screenshorts/img5.png)



