# EE 4065 - Homework 6: Handwritten Digit Recognition on STM32

![STM32](https://img.shields.io/badge/Hardware-STM32H743ZI-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20Lite-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the implementation of **Handwritten Digit Recognition** using Convolutional Neural Networks (CNN) on an **STM32** microcontroller.

This project implements the application described in **Section 13.7** of the course textbook, with significant optimizations to fit Deep Learning models into limited Flash memory.

## 👥 Team Members
 [Mehmet Mete EKER] - [150722013]
[Muhammet Emre MEMİLİ] - [150721033]

## 🎯 Project Objective
The goal is to deploy a CNN model to recognize handwritten digits (MNIST) on an embedded system. We compared three different architectures mentioned in the course:
1.  **ResNet50** (Standard Reference)
2.  **EfficientNetB0** (Efficient Architecture)
3.  **MobileNetV2** (Lightweight Architecture)

## ⚡ Technical Challenge & Solution
The STM32H743ZIT6 has **2MB of Internal Flash**. Standard transfer learning models are too large to fit.

**Our Optimization Strategy:**
1.  **Architecture Tuning:** We utilized **MobileNetV2** with a width multiplier of `alpha=0.35`. This reduced the parameter count by ~85% compared to the standard version.
2.  **Post-Training Quantization:** We converted model weights from **Float32** to **Int8**, achieving a further 4x size reduction.

### Model Size Comparison Table

| Model Architecture | Configuration | Original Size (Float32) | Optimized Size (Int8) | Deployment Status |
| :--- | :--- | :---: | :---: | :--- |
| **ResNet50** | Standard | ~98.0 MB | ~24.0 MB | ❌ Too Large |
| **EfficientNetB0** | Standard | ~16.0 MB | ~4.5 MB | ⚠️ Requires Ext. Flash |
| **MobileNetV2** | Standard (`alpha=1.0`) | ~9.5 MB | ~2.6 MB | ⚠️ Fits External Only |
| **MobileNetV2** | **Optimized (`alpha=0.35`)** | **~2.0 MB** | **~0.5 MB** | ✅ **Fits Internal Flash** |

*As a result, the **MobileNetV2 (Alpha 0.35, Int8)** model was selected for the final deployment.*

## 📂 Repository Structure

```text
├── 1_Training/                 # Python Workflow
│   ├── train_optimize.py       # Main script: Training -> Quantization -> .cc Generation
│   └── requirements.txt        # Python dependencies
├── 2_Models/                   # Generated Model Files
│   ├── stm32_final_models/     # Contains the optimized models
│   │   ├── MobileNetV2_0.35_quantized.tflite  # <--- The model used on MCU
│   │   └── MobileNetV2_0_35_data.cc           # C array for the project
├── 3_STM32_Project/            # STM32CubeIDE Project
│   ├── Core/Src/main.c         # Inference logic
│   ├── X-CUBE-AI/              # AI Library files
│   └── ...
├── Report.pdf                  # Detailed Homework Report
└── README.md
