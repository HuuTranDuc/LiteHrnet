import os
import numpy as np
import json
import sys
import torch
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

ground_truth_path = r".\data\coco\annotations\test.json"          # annotation_path
output_path = r"./output/output.json"                             # predict_path             

def plt_imshow(img, title_name=None, axis_on = False, figsize=(6, 6), save_fig_at = None, dont_show_figure = False):
    fig, ax = plt.subplots(1, 1, figsize=(figsize))
    # if title_name != None: fig.suptitle(f"{title_name}", fontsize=16)  
    if title_name != None: plt.title(f"{title_name}") 
    if axis_on:
        plt.grid()
        # pass
    else:
        ax.axis('off')
    
    plt.tight_layout()
    ax.imshow(img)
    if save_fig_at != None: 
        plt.savefig(f"{save_fig_at}" )

    if not(dont_show_figure):
        # ax.imshow(img)
        plt.show()    

def keypoint_similarity(gt_kpts, pred_kpts, sigmas, areas):
    """
    Params:
        gts_kpts: Ground-truth keypoints, Shape: [M, #kpts, 3],
                  where, M is the # of ground truth instances,
                         3 in the last dimension denotes coordinates: x,y, and visibility flag
                          
        pred_kpts: Prediction keypoints, Shape: [N, #kpts, 3]
                   where  N is the # of predicted instances,
 
        areas: Represent ground truth areas of shape: [M,]
 
    Returns:
        oks: The Object Keypoint Similarity (OKS) score tensor of shape: [M, N]
    """
     
    # epsilon to take care of div by 0 exception.
    EPSILON = torch.finfo(torch.float32).eps
     
    # Eucleidian dist squared:
    # d^2 = (x1 - x2)^2 + (y1 - y2)^2
    # Shape: (M, N, #kpts) --> [M, N, 17]
    dist_sq = (gt_kpts[:,None,:,0] - pred_kpts[...,0])**2 + (gt_kpts[:,None,:,1] - pred_kpts[...,1])**2
 
    # Boolean ground-truth visibility mask for v_i > 0. Shape: [M, #kpts] --> [M, 17]
    vis_mask = gt_kpts[..., 2].int() > 0
 
    # COCO assigns k = 2σ.
    k = 2*sigmas
 
    # Denominator in the exponent term. Shape: [M, 1, #kpts] --> [M, 1, 17]
    denom = 2 * (k**2) * (areas[:,None, None] + EPSILON)
 
    # Exponent term. Shape: [M, N, #kpts] --> [M, N, 17]
    exp_term = dist_sq / denom
 
    # Object Keypoint Similarity. Shape: (M, N)
    oks = (torch.exp(-exp_term) * vis_mask[:, None, :]).sum(-1) / (vis_mask[:, None, :].sum(-1) + EPSILON)
 
    return oks

keypoints= [
                "Nose",
                "Left Eye",
                "Right Eye",
                "Left Ear",
                "Right Ear",
                "Left Shoulder",
                "Right Shoulder",
                "Left Elbow",
                "Right Elbow",
                "Left Wrist",
                "Right Wrist",
                "Left Hip",
                "Right Hip",
                "Left Knee",
                "Right Knee",
                "Left Ankle",
                "Right Ankle"
            ]

OKS_keypoint ={
                "Nose" :[],
                "Left Eye" :[],
                "Right Eye" :[],
                "Left Ear" :[],
                "Right Ear" :[],
                "Left Shoulder" :[],
                "Right Shoulder" :[],
                "Left Elbow" :[],
                "Right Elbow" :[],
                "Left Wrist" :[],
                "Right Wrist" :[],
                "Left Hip" :[],
                "Right Hip" :[],
                "Left Knee" :[],
                "Right Knee" :[],
                "Left Ankle" :[],
                "Right Ankle" :[] 
            }

with open(ground_truth_path, "r") as f:
    ground_truth = json.load(f)
    
with open(output_path, "r") as f:
    output = json.load(f)

KPTS_OKS_SIGMAS_COCO = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

for threshold in [0.5, 0.75, 0.95]:
    
    print(f"@OKS = {threshold}")
    total_ap = 0
    for idx,keypoint in enumerate(keypoints):
        OKS = []
        Score = []
        
        for i in range(len(output)):
            gt_kpts = torch.tensor([ground_truth["annotations"][i]["keypoints"][idx*3:(idx*3+3)]], dtype = torch.float)
            det_kpts = torch.tensor([output[i]["preds"][0][idx]], dtype = torch.float)
            areas = torch.tensor([ground_truth["annotations"][i]["area"]])
            score_model = torch.tensor([output[i]["preds"][0][idx][2]])
            oks_coco = keypoint_similarity(gt_kpts.unsqueeze(0),
                                    det_kpts.unsqueeze(0),
                                    sigmas=KPTS_OKS_SIGMAS_COCO[idx],
                                    areas=areas)
            
            Score.append(score_model)
            OKS.append(oks_coco)
        y_true = np.array([1 if oks >= threshold else 0 for oks in OKS])
        y_scores = np.array([sc for sc in Score])  # Score
        
        # Precision và Recall
        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        # AP
        ap = average_precision_score(y_true, y_scores)
        total_ap += ap
        calcu_oks = (sum(OKS)/len(output)).item()
        print(f'{keypoint} mAP: {ap:.4f} - OKS: {calcu_oks:.4f}')
    print(f"total mAP {threshold} = {total_ap/17}")
    print(f"------------------------------------")