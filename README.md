# Fuji_Ripeness_And_Size_Estimation

## How-to-use

1. Create a python env
2. __pip install opencv-python numpy==1.24.1 scipy matplotlib__
3. Install pytorch: __pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117__
4. Install Segment Anything: __pip install git+https://github.com/facebookresearch/segment-anything.git__ (https://github.com/facebookresearch/segment-anything)
5. Install mmdetection: 
        __pip install -U openmim__
        __mim install mmengine__
        __mim install "mmcv==2.1.0"__
        __mim install mmdet__
6. __pip install fairscale transformers__
7. python3 inference.py (Don't forget to change the paths for image, checkpoints, etc.)
