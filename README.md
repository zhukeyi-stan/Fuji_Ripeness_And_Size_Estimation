# Fuji_Ripeness_And_Size_Estimation

## Dataset

Dataset is available at https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset

## How-to-use

1. Create a python env
2. Install packages:
   
        pip install opencv-python numpy==1.24.1 scipy matplotlib
        pip install fairscale transformers
   
4. Install pytorch:
   
        pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
   
5. Install Segment Anything (https://github.com/facebookresearch/segment-anything):

        pip install git+https://github.com/facebookresearch/segment-anything.git
   
6. Install mmdetection (https://mmdetection.readthedocs.io/en/latest/get_started.html):
   
        pip install -U openmim
        mim install mmengine
        mim install "mmcv==2.1.0"
        mim install mmdet
        
7. Run Code (Don't forget to change the paths for image, checkpoints, etc.)

        python3 inference.py 
