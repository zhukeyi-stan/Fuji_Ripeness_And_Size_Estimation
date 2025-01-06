import cv2
import argparse
import os
import numpy as np
import torch
from utils.sizing_algorithms import size_estimation

# grounding dino
from mmdet.apis import init_detector, inference_detector
# segment anything
from segment_anything import (
    sam_model_registry,
    SamPredictor
)

from utils.plot import save_fig

import time

def detect(detector,img,score_thres=0.3): # bbox_generator
    t0=time.time()
    res=inference_detector(detector,img,text_prompt='ripe. unripe.',custom_entities=True)
    pred_bboxes=res.pred_instances.bboxes[res.pred_instances.scores>score_thres,:]
    pred_labels=res.pred_instances.labels[res.pred_instances.scores>score_thres] # 1 for unripe, 0 for ripe
    pred_scores=res.pred_instances.scores[res.pred_instances.scores>score_thres]
    pred_bboxes=pred_bboxes.cpu()
    pred_labels=pred_labels.cpu()
    pred_scores=pred_scores.cpu()
    t1=time.time()
    return pred_bboxes,pred_labels,pred_scores,t1-t0

def segment(segmentor,img,pred_bboxes):
    t0=time.time()
    transformed_bboxes=segmentor.transform.apply_boxes_torch(pred_bboxes,img.shape[:2]).to(device)
    segmentor.set_image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    masks, _, _ = segmentor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_bboxes.to(device),
        multimask_output = False,
    )
    t1=time.time()
    return masks,t1-t0

##
device='cuda:0'
## GDINO
model_path='/home/stan/Projects/SH_stage1_final/Selective_Harvesting/mmdet_result/gdino_b_full.py'
ckpt_path='/home/stan/Projects/SH_stage1_final/Selective_Harvesting/mmdet_result/gdino.pth'
## SAM
sam_version = 'vit_h' #SAM ViT version: vit_b / vit_l / vit_h
sam_checkpoint = '/home/stan/Projects/SH_stage1_final/Selective_Harvesting/Grounded-Segment-Anything/weights/sam_vit_h_4b8939.pth'
sam_hq_checkpoint = '/home/stan/Projects/SH_stage1_final/Selective_Harvesting/Grounded-Segment-Anything/weights/sam_hq_vit_h.pth'
use_sam_hq = False

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--outlier-removal-low',type=int,default=30,help='Outlier Removal Rate, Lower Bound')
    parser.add_argument('--outlier-removal-high',type=int,default=70,help='Outlier Removal Rate, Higher Bound')
    parser.add_argument('--sizing-algorithm',type=str,default='max_dist_mask',help='From one of [\'largest_segment\',\'least_square\',\'bounded_ransac\',\'bbox\',\'max_dist_mask\',\'hough_mask\']')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    image_dir='/home/stan/Projects/SH_stage1_final/Selective_Harvesting/fuji/images/'
    depth_dir='/home/stan/Projects/SH_stage1_final/Selective_Harvesting/fuji/depths/'
    image_name='_MG_2657_24.png'
    # image_name='_MG_2662_10.png'
    depth_name=image_name.replace('.png','.npy')
    image_path=os.path.join(image_dir,image_name)
    depth_path=os.path.join(depth_dir,depth_name)

    # Image from UdL
    focal_length=5805.34
    # Image from Unibo
    # focal_length=1383.77

    args=parse_args()
    ## Init Detector and Segmentor
    detector=init_detector(model_path,ckpt_path,device=device)
    segmentor=SamPredictor(sam_model_registry[sam_version](checkpoint=sam_checkpoint).to(device))

    outlier_removal_rate=(args.outlier_removal_low,args.outlier_removal_high)
    sizing_algorithm=args.sizing_algorithm

    ## Inference
    img=cv2.imread(image_path)
    depth=np.load(depth_path)
    pred_bboxes,pred_labels,pred_scores,detection_time=detect(detector,img)
    if pred_bboxes.shape[0]==0:
        pass
    else:
        masks,segmentation_time=segment(segmentor,img,pred_bboxes)
        pc_pack=[]
        mask_pack=[]
        bbox_pack=[]
        d_pack=[]
        for i in range(pred_bboxes.shape[0]):
            bbox=pred_bboxes[i,:].cpu().numpy()
            mask=masks[i,0,:,:].cpu().numpy()
            if sizing_algorithm in ['largest_segment','least_square','bounded_ransac']:
                args=(img,depth,bbox,mask,focal_length,outlier_removal_rate)
                d=size_estimation(sizing_algorithm,args)
            elif sizing_algorithm in ['bbox','max_dist_mask','hough_mask']:
                args=(depth,mask,bbox,focal_length,outlier_removal_rate)
                d=size_estimation(sizing_algorithm,args)
            d_pack.append(d)
    ## Post Process
    save_fig(cv2.cvtColor(img,cv2.COLOR_RGB2BGR),pred_bboxes,masks,pred_labels,'Inference_Results.jpg',d_pack)
    pass