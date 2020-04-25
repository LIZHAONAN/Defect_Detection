# Defect Detection Pipeline

Pipeline for positive and negative defect detection in active nematics.
This repository includes code for detection and visualization.

<p float='left'>
    <img src="temp/defects_confined.jpg" style="float:left" width=300px>
    <img src="temp/defects_unconfined.jpg" style="float:right" width=300px>
</p>

Improvements from previous implementation:
- Support for multiple GPUs and CPU
- Optimized custom data loader for detection
- Visualization tools
- Organized and documented code

## Getting started
Install dependencies with pip:
```
pip install -r requirements.txt
```

Download pre-trained weights [here](https://drive.google.com/drive/folders/1FE28rAh88YuCtu8LZ7Zh_yY_yYcJ5hRP?usp=sharing)

## Detection
The detection has two stages. We first use YOLO to quickly
select possible regions of detects, and then use two ResNet scanners to examine
the proposed regions. If no YOLO result is given, the scanners will examine
every pixel in the image (but may take much longer time).

0. Normalize input data set. This can be done by setting ```mean``` and 
    ```std``` in ```utils.py```. Set ```mean``` to 0 and ```std``` to 1 
    if the input data set is already normalized.
    
1. To specify input images, create a .csv file with one column.
In the given example, ```img_paths.csv``` consists of two image paths:
    ```csv
    path
    images/confined.jpg
    images/unconfined.jpg
    ```
2. (Optional) Run YOLO model to quickly propose regions with high probability,
    save results to ```yolo.csv```
    ```bash
    python yolo_detect.py --pth models/yolo.pth \
                          --images img_paths.csv \
                          --output yolo.csv
    ```
3. (Optional) Visualize YOLO results, save output in the current directory
    ```bash
    python visualize.py --yolo yolo.csv --save .
    ```
    <p float='left'>
        <img src="temp/yolo_confined.jpg" style="float:left" width=300px>
        <img src="temp/yolo_unconfined.jpg" style="float:right" width=300px>
    </p>
4.  Run defect detection, save results to ```defects.csv```
    ```bash
    python detect.py --hard models/res18_hard.pth \
                     --uniform models/res34_uniform.pth \
                     --integrator models/integrator.pth \
                     --images img_paths.csv \
                     --yolo yolo.csv \
                     --output defects.csv
    ```
5. (Optional) Visualize defects, save output in the current directory
    ```bash
    python visualize.py --defects defects.csv --save .
    ```
    