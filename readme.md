# CS523- Reimplement of YoloV3

## YoloV3
---------------------------------------------------------------------------
Jianxiao Yang  
yangjx@bu.edu  
U77864846

### Requirements
---------------------------------------------------------------------------
input size: 3 x 416 x 416  
please run the following command to install the required packages:  
`pip install -r requirements.txt`  
### datasets
---------------------------------------------------------------------------
We use the VOC2007 dataset for training and testing.  
dataset link: [here](https://drive.google.com/file/d/1Q5__3aoS56xpg00HGpck0_6o_jL78km8/view?usp=share_link).  
unzip that in your root directory, and run `python voc_annotation.py` to generate the training data.

### How to compile and run
---------------------------------------------------------------------------
For trainning: please first define your classes in the 'model_data' directory and put the dataset under root directory.    
Change the path root and parameters in train.py.  
To fine tune the model, please set unfrozen epochs the same as frozen epochs.  
run `python train.py` to train the model.  
If you have multiple gpus, run `CUDA_VISIBLE_DEVICES=0,1 python train.py` to train the model on multiple gpu.  

The trained model will be saved in 'logs/' directory.  

For predicting: put all images in './img' and run `python predict.py` to predict the images. The results will be stored in "./img_out"  

### Reference
---------------------------------------------------------------------------
[https://github.com/bubbliiiing/yolo3-pytorch.git](https://github.com/bubbliiiing/yolo3-pytorch.git)