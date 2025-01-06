---
title: "Fractured Scaphoid Detection"
date: 2025-01-07T01:00:59+08:00
tags: ['Detection', 'Faster R-CNN', 'YOLOv11-OBB']
summary: "Use Faster [R-CNN](https://arxiv.org/abs/1506.01497) and YOLOv11-[OBB](https://docs.ultralytics.com/datasets/obb/) to detect the scaphoid fracture location. "
---


## Get started

1. Training 
    ```python
    python main.py --train 1
    ```
2. Run System
    
    ```python
    python main.py
    ```
    

## Model

| Name | Description | path |
| --- | --- | --- |
| **ScaphoidDetector** | Detects scaphoid bone in X-ray hand images using [Faster R-CNN](https://arxiv.org/abs/1506.01497) | `scaphoid_detector.py` |
| **FractureClassifier** | Classify scaphoid fractures using [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html) pre-trained model after detection by ScaphoidDetector | `fracture_classifier.py` |
| **HandDetector** | Detects scaphoid bones and fractures region in X-ray hand image using YOLOv11-[OBB](https://docs.ultralytics.com/datasets/obb/) | `hand_detector.py` |

## Methods

1. ScaphoidDetector + FractureClassifier + HandDetector
    
    First, use Faster R-CNN to detect the scaphoid bone in the full X-ray hand image. Then, use VGG16 to classify whether there is a fracture. Finally, use YOLOv11-obb to detect the fracture location.
    
2. HandDetector
    
    Directly use YOLOv11-obb to detect the scaphoid bone and fracture locations.
    

## **ScaphoidDetector + FractureClassifier + HandDetector**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/527516ba-ec1d-4333-b649-e193cba1a90d/image.png)

### Datasets

1. File Structure: 
    ```python
    ip_data  
    ├── fracture_detection  
    │   └── annotations // Fracture locations: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  
    └── scaphoid_detection  
        ├── annotations // Scaphoid locations: [x1, y1, x2, y2]  
        └── images      // Hand X-ray images  
    ```

1. After data preprocessing in `dataset.py` :
    
    `all_datas.json` and new folders will be created under fracture_detection and scaphoid_detection
    
    ```python
    ├── ip_data
    	  ├── fracture_detection
    	  │   ├── annotations
    	  │   ├── images
    	  │   └── images_rec
    	  └── scaphoid_detection
    	      ├── annotations
    	      ├── images
    	      └── images_rec
    ├── all_datas.json
    ```
    
    - `fracture_detection/` :
        - `images/` : Contains the full scaphoid images cropped based on scaphoid locations.
        - `images_rec/` : Contains the scaphoid images with highlighted fracture locations.
        
        ```python
        fracture_detection
        ├── annotations
        ├── images
        └── images_rec
        ```
        
    - `scaphoid_detection/images_rec` : Stores hand images with the scaphoid region framed.

### Training

1. Train ScaogiudDetector
    
    ```python
    from scahpoid_detector import ScaphoidDetector
    scaphoid_detector = ScaphoidDetector(args)
    scaphoid_detector.train()
    ```
    
2. Train FractureClassifier
    
    ```python
    from fracture_classifier import FractureClassifier
    fracture_classifier = FractureClassifier(args)
    fracture_classifier.train()
    ```
    
3. Train HandDetector
    
    ```python
    from hand_detector import HandDetector
    hand_detector = HandDetector(args)
    hand_detector.train()
    ```
    
4. Analysis
    - ScaphoidDetector
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/ce8bf7df-5077-4532-a170-47769001caa1/image.png)
        
    - FractureClassifier
        
        accuracy, recalls, precision, f1, loss
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/a306967d-9e07-4485-9ad4-75645480b863/image.png)
        
    - HandDetector: Curves will be saved in `performance` and  `runs/`  respectively
        
        ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/94f401d1-061e-4e91-997a-f3bb1a421ffe/image.png)
        

### Detecting

Steps 1. Detect Scaphoid

- Use `detect()` function
    
    ```python
    scaphoid_detector.detect(dir_path)
    ```
    
- Detected scaphoid location will be cropped and saved in `prediction/scaphoid/`
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/0d7062e3-c432-488b-bbae-c2fd86ea73b1/image.png)
    

Steps 2. Classify fracture

- Use `classify()` function
    
    ```python
    fracture_classifier.classify(dir_path)
    ```
    
- Fracture scaphoid will be saved in `prediction/classifier/`

Steps 3. Detect fracture location

- Use `detect_fracture()` function
- The images with marked fracture locations will be saved in `prediction/fracture/`
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/d4cb9cf1-1573-4c54-9f86-638706d189d8/image.png)
    

## HandDetector

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/2f2a1611-db52-4026-b225-7ac49cd84c2d/image.png)

### Training Datasets

Use functions from `yolo_anno.py` to construct data for YOLOv11-OBB
1. File Structure
    ```python
    yolo_config
    ├── data
    ├── datasets
    │   ├── fracture
    │   │   ├── images
    │   │   │   ├── train
    │   │   │   └── val
    │   │   └── labels
    │   │       ├── train
    │   │       └── val
    │   └── hand
    │       ├── images
    │       │   ├── train
    │       │   └── val
    │       └── labels
    │           ├── train
    │           └── val
    └── weights
    ```

2. During Training: YOLO 會自動將所有圖片拼在一起，最後再裁成設定得大小 (以下範例為1024)，圖片就會前處理成以下，一個batch的圖片數量會根據 `batch_size` (以下範例為 8)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/24c73464-a8e2-4fe5-93b6-72811075c6c6/image.png)
    

### Training

1. Train HandDetector
    ```python
    from hand_detector import HandDetector
    hand_detector = HandDetector(args)
    hand_detector.train()
    ```

2. Results will be saved in `runs/`
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/aa14525d-8d76-4d12-b39a-501fcb275de6/image.png)
    

### Results

1. Confusion Matrix: 
    - **Scaphoid:** Using YOLOv11-OBB to detect the position of the scaphoid performed exceptionally well, with an accuracy of up to 98% in predictions.
    - **Fracture:** YOLOv11-OBB correctly predicted 41% of fracture locations in full-hand X-ray images, slightly outperforming the two-stage detection method.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/6977b654-bf48-4842-8962-e0710b1fc121/image.png)
    
2. Precision, Recall, F1
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/e0885c3a-c85d-45a6-8aa8-002dcd4a99e5/image.png)
    
3. During Testing
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/0198ca53-98df-4f3f-9013-e0822b20fecd/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/b8847e82-05a5-4d3e-9ed2-617656c16d50/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/8e42960a-62b0-48b0-83f7-818cc3391d61/image.png)
    

### Detecting

1. Detect scaphoid
    - Detect images in folder
        
        ```python
        hand_detector.detect_scaphoid(dir_name)
        ```
        
    - Detect one image
        
        ```python
        hand_detector._detect_scaphoid(img_name, img_path)
        ```
        
2. Detect fracture
    - Detect images in folder
        
        ```python
        hand_detector.detect_fracture(dir_name)
        ```
        
    - Detect one image
        
        ```python
        hand_detector._detect_fracture(img_name, img_path)
        ```
        
3. Plot the rectangle
    
    The `detect_*()` function performs two key operations:
    
    - Predicts the location of the scaphoid or fracture
    - Uses `plot_xyxyxyxy()` to visualize the results with
        - Red rectangles showing the target (ground truth) locations
        - Green rectangles showing the predicted locations
        - Pictures will be saved in `prediction/hand/`
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/66b9bca3-49ed-433d-97b7-6e010625718c/image.png)
    

## System

Load a folder containing the dataset file structure. The system will then begin predicting and save the images with the scaphoid and fracture locations highlighted.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/4b659a84-7683-4662-bbbf-bf74900d1d81/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5c0abaf5-b9fa-4a9b-97e3-5ae6c274eec7/41af7a37-a41e-4a88-a40b-0332b21f98e5/image.png)

## Code Availability

https://github.com/Hlunlun/Fractured-Scaphoid-Detection

## Datasets Availability

From [NCKU CSIE Visual System Lab](https://sites.google.com/view/ncku-csie-vslab/home)

## Reference

- [**FastRCNNPredictor doesn't return prediction in evaluation**](https://github.com/pytorch/vision/issues/1952)
- [**Oriented Bounding Box (OBB) Datasets Overview**](https://docs.ultralytics.com/datasets/obb/)
- [**一篇文章快速认识YOLO11 | 旋转目标检测 | 原理分析 | 模型训练 | 模型推理**](https://blog.csdn.net/qq_41204464/article/details/143217068)
- [**Understanding and Implementing Faster R-CNN**](https://medium.com/@RobuRishabh/understanding-and-implementing-faster-r-cnn-248f7b25ff96)
- [**The Detection and Classification of Scaphoid Fractures in Radiograph by Using a Convolutional Neural Network**](https://www.mdpi.com/2075-4418/14/21/2425)
- [**yolov5_obb: A comprehensive tutorial from data preparation to model deployment**](https://medium.com/@CVHub520/yolov5-obb-a-comprehensive-tutorial-from-data-preparation-to-model-deployment-8d7c6a98388f)
- [PolygonObjectDetection](https://github.com/XinzeLee/PolygonObjectDetection)
- [**How to use YOLOv11 for Object Detection**](https://medium.com/@Mert.A/how-to-use-yolov11-for-object-detection-924aa18ac86f)
