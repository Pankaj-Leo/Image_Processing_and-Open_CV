# 🧠 Computer Vision Concepts and Projects

*A hands-on portfolio exploring computer vision fundamentals and deep learning applications with OpenCV, CNNs, and Transfer Learning.*

---

## 🌍 Overview

This repository is a **comprehensive learning suite** covering the complete journey of computer vision — from basic **image operations using OpenCV** to **advanced CNN architectures** like VGG16 and AlexNet.

Each module focuses on a practical aspect of image understanding, transformation, and recognition, with detailed Jupyter notebooks and reference PDFs.

---

## 🗂️ Repository Structure

```
Computer vision concepts and projects/
│
├── Open CV/
│   ├── Working_with_video_files/
│   ├── Affine_&_Perspective_Transformation/
│   ├── Edge_Detection/
│   ├── Haar Cascade/
│   ├── Flip_Rotate_Crop_Images/
│   ├── Image_Resizing_Scaling_interpolation/
│   ├── Histogram_Equalization/
│   ├── CLAHE/
│   ├── Color_Thresholding/
│   ├── Image_Segmentation_Using_opencv/
│   ├── Applying_blur_filters/
│   ├── Exploring_Color_space/
│   ├── Image_filters/
│   ├── contours/
│   ├── Calculating_and_plotting_histogram/
│   ├── Adding_text_to_images/
│
├── Image_Augmentation/
│
├── Apple_Stock_Price_Prediction/  ← Demonstrates CNNs applied to time-series and finance
│
├── Advanced-CNN-Architectures-master/
│   ├── Transfer Learning VGG16.ipynb
│   ├── Transfer Learning Alexnet.ipynb
│   ├── README.md
│
└── README.md (this master file)
```

---

## 🧩 Core Learning Modules

### 1️⃣ OpenCV Essentials — Image Operations
Fundamentals of image handling and preprocessing using **OpenCV**.

**Covered Topics**
- Reading, writing, and displaying images (`cv2.imread`, `cv2.imshow`)
- Color spaces (RGB, HSV, LAB, Grayscale)
- Image resizing, rotation, translation, scaling
- Edge detection (Sobel, Canny, Laplacian filters)
- Histogram equalization and CLAHE
- Contours, blurring, thresholding, and text overlays

**Key Folders**
- `Edge_Detection/`
- `Histogram_Equalization/`
- `CLAHE/`
- `Image_filters/`
- `Contours/`

**Reference PDFs**
- `014. Edge Detection Using Sobel, Canny & Laplacian.pdf`
- `015. Calculating and Plotting Histograms.pdf`
- `013. Applying Blur Filters – Average, Gaussian, Median.pdf`

---

### 2️⃣ Image Segmentation using OpenCV
Implements various segmentation techniques using **color thresholding**, **region-based segmentation**, and **k-means clustering**.

**Key Notebook:** `019. Image Segmentation Using OpenCV.pdf`  
**Concepts:** 
- Binary segmentation
- Mask creation using cv2.inRange
- Region-growing and connected components
- Morphological operations (Erosion, Dilation, Opening, Closing)

---

### 3️⃣ Object Detection — Haar Cascades
Implements **face detection and object localization** using Haar Cascade classifiers.

**Key File:** `020. Haar Cascade for Face Detection.pdf`  
**Concepts:**
- Face and eye detection using pretrained Haar classifiers  
- Bounding box visualization using OpenCV  
- Real-time detection using webcam streams  

---

### 4️⃣ Advanced CNN Architectures — Transfer Learning

This section explores **transfer learning** using pretrained architectures like **AlexNet** and **VGG16**.

**Files**
- `Transfer Learning VGG16.ipynb`
- `Transfer Learning AlexNet.ipynb`

**Concepts**
- Fine-tuning pretrained models for custom datasets  
- Feature extraction vs full fine-tuning  
- Comparative analysis between AlexNet and VGG16  

---

### 5️⃣ Image Augmentation
Enhancing dataset diversity through transformations for better model generalization.

**Techniques Used**
- Random rotations, flips, zooms, and brightness adjustments  
- Implemented using OpenCV and Keras ImageDataGenerator  
- Augmentation visualization for dataset inspection  

---

### 6️⃣ Apple Stock Price Prediction (Bonus Project)
Applies **CNNs and RNNs** to predict Apple’s stock prices — demonstrating how **computer vision pipelines** and **temporal modeling** can merge.

**Concepts**
- Data preprocessing with normalization  
- CNNs for feature extraction from time windows  
- LSTM/GRU for sequence prediction  
- Performance visualization (MAE, RMSE)

---

## 📘 Learning Path Summary

| Module | Focus | Tools & Libraries |
|--------|-------|-------------------|
| OpenCV Basics | Image operations & filtering | OpenCV, NumPy, Matplotlib |
| Segmentation | Thresholding, clustering | OpenCV, Scikit-image |
| Object Detection | Haar cascades | OpenCV |
| Transfer Learning | AlexNet, VGG16 | PyTorch, TensorFlow, Keras |
| Augmentation | Data preprocessing | Albumentations, Keras |
| Stock Prediction | Vision + Finance | Pandas, PyTorch/TensorFlow |

---

## ⚙️ Requirements

Install dependencies before running notebooks:

```bash
pip install opencv-python numpy matplotlib tensorflow keras torch torchvision albumentations scikit-learn pandas
```

---

## 🚀 Example Runs

### Run an OpenCV demo
```bash
cd Open\ CV/Image_Segmentation_Using_opencv
jupyter notebook 019.\ Image\ Segmentation\ Using\ OpenCV.ipynb
```

### Run Transfer Learning
```bash
cd Advanced-CNN-Architectures-master
jupyter notebook Transfer\ Learning\ VGG16.ipynb
```

---

## 🧠 Key Learning Outcomes

By completing this repository, you will:
- Master **OpenCV** operations for real-world image manipulation  
- Understand **image segmentation and thresholding** techniques  
- Implement **object detection using classical and CNN methods**  
- Learn **transfer learning** with pretrained CNNs  
- Apply **augmentation and preprocessing** techniques effectively  
- Gain exposure to **cross-domain AI applications** (vision + finance)

---

Special Thanks to Krish Naik (https://github.com/krishnaik06)

---
**Pankaj Somkuwar**  
🔗 [GitHub](https://github.com/Pankaj-Leo) | [LinkedIn](https://linkedin.com/in/pankajsomkuwar)

---

## 🏁 License  
Released under the **MIT License** — open for learning, research, and development.

---

