import argparse
import datetime
import os
import sys
import socket
hostname = socket.gethostname()
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '../'))
sys.path.append(os.path.join(here))

from misc.print_utils import *
from misc.other_utils import *
from misc.rle_tools import encode_RLE, decode_RLE

import itertools
import multiprocessing as mp
import torch.multiprocessing as mp
import os
import time
import warnings
from argparse import ArgumentParser
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import List, Optional, Sequence, Union

import cv2
import json_tricks as json
import numpy as np
import torch
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import torch.nn as nn
import torch.nn.functional as F
from sapiens_liter.liter.utils.pose_utils import nms, top_down_affine_transform, udp_decode, udp_decode_torch

from tqdm import tqdm

from ultralytics import YOLO

from torchvision.transforms.functional import crop
from torchvision.transforms.functional import resize
from PIL import Image

def crop_resize_and_normalize(orig_img, bboxes_list, sam2_masks_list, sapiens_original_input_shape, mean, std, draw_boxes = True, show_bboxes = False):
    """Preprocess pose images and bboxes."""
    preprocessed_images = []
    extended_padded_bboxes = []
    # cropped_images = []
    # Create a new element for each box

    for bbox_idx, bbox in enumerate(bboxes_list):    
        
        sam2_mask = sam2_masks_list[bbox_idx]
            
        # Set the padding ratio
        pad_ratio = 1.25
        desired_aspect_ratio = sapiens_original_input_shape[1] / sapiens_original_input_shape[0] # width / height of the model

        # Calculate the padded bbox
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        padded_box_width = width * pad_ratio
        padded_box_height = height * pad_ratio
        padded_bbox = np.array([x_center - (padded_box_width / 2), y_center - (padded_box_height / 2), x_center + (padded_box_width / 2), y_center + (padded_box_height / 2)])
        
        # Calculate the extended padded bbox with the desired aspect ratio
        if (padded_box_width / padded_box_height) > desired_aspect_ratio:
            extended_padded_box_height = padded_box_width / desired_aspect_ratio
            extended_padded_box_width = padded_box_width
        else:
            extended_padded_box_width = padded_box_height * desired_aspect_ratio
            extended_padded_box_height = padded_box_height
            
        extended_padded_bbox = np.array([x_center - (extended_padded_box_width / 2), y_center - (extended_padded_box_height / 2), x_center + (extended_padded_box_width / 2), y_center + (extended_padded_box_height / 2)])
        
        # Store the extended padded bbox
        extended_padded_bboxes.append(extended_padded_bbox)
        
        
        # Crop, resize and normalize the image
        orig_img_tensor = torch.from_numpy(orig_img).permute(2, 0, 1).clone() # C, H, W
        orig_img_tensor[:, sam2_mask == 0] = 0 # set the background to 0
        cropped_img = crop(orig_img_tensor, int(extended_padded_bbox[1]), int(extended_padded_bbox[0]), int(extended_padded_bbox[3] - extended_padded_bbox[1]), int(extended_padded_bbox[2] - extended_padded_bbox[0])) # shape: (C, H, W)
        cropped_img_resized = resize(cropped_img, (sapiens_original_input_shape[0], sapiens_original_input_shape[1]), interpolation=Image.BILINEAR) # shape: (C, H, W)
        cropped_img_resized = cropped_img_resized.float()
        mean = torch.Tensor(mean).view(-1, 1, 1)
        std = torch.Tensor(std).view(-1, 1, 1)
        cropped_img_resized = (cropped_img_resized - mean) / std
        
        # Store the preprocessed image
        preprocessed_images.append(cropped_img_resized)
        
        # Draw the bboxes on the image and show it
        if draw_boxes or show_bboxes:
            cv2.rectangle(orig_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(orig_img, "original bbox", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(orig_img, (int(padded_bbox[0]), int(padded_bbox[1])), (int(padded_bbox[2]), int(padded_bbox[3])), (0, 255, 255), 2)
            cv2.putText(orig_img, "padded bbox", (int(padded_bbox[0]), int(padded_bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.rectangle(orig_img, (int(extended_padded_bbox[0]), int(extended_padded_bbox[1])), (int(extended_padded_bbox[2]), int(extended_padded_bbox[3])), (0, 255, 0), 2)
            cv2.putText(orig_img, "padded bbox with desired ratio", (int(extended_padded_bbox[0]), int(extended_padded_bbox[1]+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cropped_img_resized_np = cropped_img_resized.numpy().transpose(1, 2, 0)

            if show_bboxes:
                cv2.imshow("Image with bboxes (type any key to continue)", orig_img)        
                key = cv2.waitKey(0)
                # if key == ord('q') or key == 27:
                #     exit()
                cv2.destroyAllWindows()
                
                cv2.imshow("Cropped, resized and normalized image (type any key to continue)", cropped_img_resized_np)
                key = cv2.waitKey(0)
                # if key == ord('q') or key == 27:
                #     exit()
                cv2.destroyAllWindows()

        
    return preprocessed_images, extended_padded_bboxes


def batch_inference_topdown(
    model: nn.Module,
    imgs: List[Union[np.ndarray, str]],
    flip=False,
    half=False,
):
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        
        
        # big_img = np.zeros((imgs[0].shape[1], imgs[0].shape[2]*len(imgs), 3), dtype=np.float32)
        # for k in range(len(imgs)):
        #     img_cpu = imgs[k].cpu().numpy().transpose(1, 2, 0)
        #     big_img[:, k*img_cpu.shape[1]:(k+1)*img_cpu.shape[1], :] = img_cpu

        # cv2.imshow("Batch image", big_img)
        # key = cv2.waitKey(0)
        # if key == ord('q') or key == 27:
        #     exit()
        
        if half:
            imgs = imgs.half()
            
        heatmaps = model(imgs.cuda())
        
        if flip:
            heatmaps_ = model(imgs.cuda().flip(-1))
            heatmaps = (heatmaps + heatmaps_) * 0.5
            
        # imgs.float().cpu()
        
    return heatmaps.float() #.cpu()


def fake_pad_images_to_batchsize(imgs, batch_size):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, batch_size - imgs.shape[0]), value=0)


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()
    
def main(args, ep, pose_estimator):

    sapiens_original_input_shape = (3, 1024, 768) # C, H, W

    image_dir = os.path.join(args.dir, "episodes", ep, "images_360")
    image_files = os.listdir(image_dir)
    image_files = [os.path.join(image_dir, f) for f in image_files if f.endswith(".jpg")]
    image_files.sort()
    
    detections_path = os.path.join(args.dir, "episodes", ep, "detections_sam2_360.json")
    if not os.path.exists(detections_path):
        raise ValueError("Detection file {} does not exist".format(detections_path))
    
    detections_dict = read_json_to_dic(detections_path)
     
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    detections_path_save = detections_path.replace(".json", f"_copy_{timestamp}.json")
    write_dic_to_json_file(detections_dict, detections_path_save)
    prInfo(f"Saved detections copy to {detections_path_save}")
    
    new_detections_dict = []
            
    for image_idx in tqdm(range(0, len(detections_dict), args.batch_size), desc="Extracting poses"):
        
        bboxes_batch = []
        sam2_masks_batch = []
        valid_det_idxs_batch = []

        # Process image 1 by 1, use the batch dimension enventually for multiple detections in one image
        valid_images_len = 0
        
        batch_images_files = [] # store the original image files
        batch_new_detections_dict_idx = [] # list of the indices of the detections object inside the new_detections_dict
        
        images_batch = []
        
        for sample_idx in range(args.batch_size):
            
            # break if the image index is out of bounds
            if image_idx + sample_idx >= len(detections_dict):
                break
            else:
                valid_images_len += 1

            image_file = detections_dict[image_idx + sample_idx]["image_file"]
            #image_shape = detections_dict[image_idx + sample_idx]["image_shape"]
            batch_images_files.append(image_file)
            

            image_path = os.path.join(args.dir, "episodes", ep, "images_360", image_file)
            assert os.path.exists(image_path), f"Image path {image_path} does not exist"
            image = np.array(cv2.imread(image_path)) # H, W, C
            images_batch.append(image)
            orig_img_shape = image.shape # H W C
            

            detections = detections_dict[image_idx + sample_idx]["detections"] # preloaded detections
            # masks_shape = detections_dict[image_idx + sample_idx]["image_shape"][:2]
            masks_shape = (image.shape[0], image.shape[1])

            new_current_detections = {"image_file": image_file, "detections": detections} # "image_shape": image_shape} # for now just a copy

            new_detections_dict.append(new_current_detections)
            batch_new_detections_dict_idx.append(image_idx + sample_idx)
        

            sample_boxes = [] # the boxes that will be used for the crops during inference for each detection of the current image
            valid_det_idxs = []  # the indexes of the valid detections in the current image
            sam2_masks = [] # the masks that will be used for the crops during inference for each detection of the current image
            for idx, det in enumerate(detections):
            
                box = [det["xmin"], det["ymin"], det["xmax"], det["ymax"]]
                width = box[2] - box[0]
                height = box[3] - box[1]
                if width > 50.0 and height > 100.0:
                    # box big enough
                    sample_boxes.append(box) # save the box
                    valid_det_idxs.append(idx) # save the index of the box
                    
                    if args.sam2_masks:
                        sam2_mask_rle = det["mask_rle"]
                        sam2_mask_rle = sam2_mask_rle.split(",")
                        sam2_mask_rle = torch.tensor([int(x) for x in sam2_mask_rle]).to(DEVICE)
                        sam2_mask = decode_RLE(sam2_mask_rle, masks_shape).cpu().numpy().astype(np.uint8) * 255
                        sam2_mask = np.array(sam2_mask) # H, W (type uint8)
                        sam2_masks.append(sam2_mask)
                    else:
                        sam2_masks.append(np.ones((orig_img_shape[0], orig_img_shape[1]), dtype=np.uint8)*255) # fake mask, H, W
                    
                
            bboxes_batch.append(sample_boxes) # list of the list of boxes for each image of the batch len(bboxes_batch) = BATCH_SIZE and bboxes_batch = [[box0, box1, box4], [box1, box4], [box1, box4], [], ...] for example where each box is a list of 4 elements [xmin, ymin, xmax, ymax]
            sam2_masks_batch.append(sam2_masks) # list of the list of masks for each image of the batch len(sam2_masks_batch) = BATCH_SIZE and sam2_masks_batch = [[mask0, mask1, mask4], [mask1, mask4], [mask1, mask4], [], ...] for example where each mask is a H, W array of type uint8
            valid_det_idxs_batch.append(valid_det_idxs) # list of the indexes (ie index inside the detections list per image) of valid detections for each image of the batch, len(valid_det_idxs_batch) = BATCH_SIZE and valid_det_idxs_batch = [[0,1,4], [1,4], [1,4], [], ...] for example
            
        ### Store the number of detections in each sample and add fake empty detections for the samples that have no detections ###
        img_bbox_map = {} # dict storing the number of detections in each sample of the batch for example : {0:3, 1:2, 2:2, 3:0, ...}, but add a fake bbox for the empty samples -> {0:1, 1:2, 2:2, 3:1, ...}
        has_dets = [] # list of booleans for each sample of the batch indicating if there are real detections in the sample, len(has_dets) = BATCH_SIZE and has_dets = [True, True, True, False, ...] for example
        for i, bboxes in enumerate(bboxes_batch):
            if len(bboxes) == 0:
                img_bbox_map[i] = 1 # one for the fake bbox
                has_dets.append(False)
                bboxes_batch[i] = np.array(
                    [[0, 0, orig_img_shape[0], orig_img_shape[1]]] # NB orig_img_shape: B H W C
                )
                sam2_masks_batch[i] = [np.zeros((orig_img_shape[0], orig_img_shape[1]), dtype=np.uint8)] # fake mask, H, W
            else:
                img_bbox_map[i] = len(bboxes)
                has_dets.append(True)
        
        
        if any(has_dets):
            pass
        else:
            # prDebug(f"Skipping batch because no valid detections")
            continue
        
        ### Preprocess the images for the inference ###
        pose_imgs, pose_imgs_extended_padded_bboxes = [], []
        for k in range(len(bboxes_batch)):
            # for each element of the batch
            preprocessed_images, extended_padded_bboxes = crop_resize_and_normalize(images_batch[k], bboxes_batch[k], sam2_masks_batch[k], (sapiens_original_input_shape[1], sapiens_original_input_shape[2]), 
                                                                                    [123.5, 116.5, 103.5], [58.5, 57.0, 57.5],
                                                                                    draw_boxes=False, show_bboxes=False)
            
            
            pose_imgs.extend(preprocessed_images)
            pose_imgs_extended_padded_bboxes.extend(extended_padded_bboxes)
        
        ### Compute the number of batches to run for the inference, because of the multiple detections in each sample ###
        n_pose_batches = (len(pose_imgs) + args.batch_size - 1) // args.batch_size

        ### Run the inference ###
        torch.compiler.cudagraph_mark_step_begin() # use this to tell torch compiler the start of model invocation as in 'flip' mode the tensor output is overwritten
        pose_results = [] # store results instead of heatmaps to save memory
        for i in range(n_pose_batches):
            imgs = torch.stack(
                pose_imgs[i * args.batch_size : (i + 1) * args.batch_size], dim=0
            )
            
            
            valid_len = len(imgs)
            imgs = fake_pad_images_to_batchsize(imgs, args.batch_size)
            heatmaps = batch_inference_topdown(pose_estimator, imgs, flip=False, half=True)[:valid_len] # weirdly flip has some very weird effect, see image 1749041540_0344029952 in episode 1 of 04-06-2025
            
            results_pairs = []
            for i in range(len(heatmaps)):
                result_pair = udp_decode_torch(
                    heatmaps[i].float(),
                    (sapiens_original_input_shape[2], sapiens_original_input_shape[1]),
                    (int(sapiens_original_input_shape[2] / 4), int(sapiens_original_input_shape[1] / 4)),
                )
                results_pairs.append((result_pair[0].cpu().numpy(), result_pair[1].cpu().numpy()))
            pose_results.extend(results_pairs)
                    
        ### Store the results in a dictionary for the valid ones ###
        batched_results = [] # list of dicts with keys heatmaps, extended_padded_bboxes, has_dets, one dict per image of the batch even the images that are empty (but they have a fake bbox): [{}, {}, {}, ...]
        for idx, (_, bbox_len) in enumerate(img_bbox_map.items()):            
            result = {
                "result_pairs": pose_results[:bbox_len].copy(), # list of results pair (keypoints, scores)
                "extended_padded_bboxes": pose_imgs_extended_padded_bboxes[:bbox_len].copy(),
                "has_dets": has_dets[idx]
            }
            batched_results.append(result)
            del (
                pose_results[:bbox_len],
                pose_imgs_extended_padded_bboxes[:bbox_len],
            )
            
        assert len(batched_results) == len(images_batch), "Error : len(batched_results) != len(images_batch), should have one dict per image and got ({} vs {})".format(len(batched_results), len(images_batch))        
        ### Post-process and save the results ###
        for image_idx_in_batch, (image, img_result, img_name) in enumerate(zip(
            images_batch[:valid_images_len], # image
            batched_results[:valid_images_len], # img_result
            batch_images_files # img_name
        )):
            
            associated_det_idxs = valid_det_idxs_batch[image_idx_in_batch] # list of the indices of the detections in the image that are being used
            current_new_detections_dict_idx = batch_new_detections_dict_idx[image_idx_in_batch]
            has_dets = img_result["has_dets"]
            
            if not has_dets:
                continue
            
            # check by image file
            image_file_in_new_detections_dict = new_detections_dict[current_new_detections_dict_idx]["image_file"]
            assert(os.path.basename(image_file_in_new_detections_dict) == os.path.basename(img_name)), "Error : image file in new detections dict {} is not the same as the image file in the batch {}".format(image_file_in_new_detections_dict, img_name)
            
            for i in range(len(img_result["result_pairs"])):
                result = img_result["result_pairs"][i] #.clone()
                # result = udp_decode(
                #     heatmap.cpu().unsqueeze(0).float().data[0].numpy(),
                #     (sapiens_original_input_shape[2], sapiens_original_input_shape[1]),
                #     (int(sapiens_original_input_shape[2] / 4), int(sapiens_original_input_shape[1] / 4)),
                # )
                # result = udp_decode_torch(
                #     heatmap.float(),
                #     (sapiens_original_input_shape[2], sapiens_original_input_shape[1]),
                #     (int(sapiens_original_input_shape[2] / 4), int(sapiens_original_input_shape[1] / 4)),
                # )

                keypoints, keypoint_scores = result # shape (1, 308, 2) and (1, 308)

                # remap keypoints to their image coordinates
                image_bbox = img_result["extended_padded_bboxes"][i].copy() # xmin, ymin, xmax, ymax in original image coordinates
                image_bbox_width = image_bbox[2] - image_bbox[0]
                image_bbox_height = image_bbox[3] - image_bbox[1]
                bbox_size = np.array([image_bbox_width, image_bbox_height])
                
                keypoints = (keypoints / (sapiens_original_input_shape[2], sapiens_original_input_shape[1])) * bbox_size + image_bbox[:2]
                keypoints = keypoints[0] # shape (308, 2)
                keypoint_scores = keypoint_scores[0] # shape (308)
                if not img_result["has_dets"]:
                    keypoint_scores = keypoint_scores*0.0 # set the scores to 0 for the samples that have no detections
                
                key_name_sapiens = f"_sapiens_{args.num_keypoints}_{args.model}"
                new_detections_dict[current_new_detections_dict_idx]["detections"][associated_det_idxs[i]]["keypoints" + key_name_sapiens] = keypoints.astype(np.int32).flatten().tolist() # list of 308x2 elements
                new_detections_dict[current_new_detections_dict_idx]["detections"][associated_det_idxs[i]]["scores" + key_name_sapiens] = [np.round(float(score), 2) for score in keypoint_scores.tolist()] # list of 308 floats
    
    write_dic_to_json_file(new_detections_dict, detections_path)
    prInfo(f"Saved detections with keypoints to {detections_path}")


def processing_loop(args, episodes_list, pose_estimator):
    """
    Process multiple episodes using the same pose estimator model.
    """
    for ep in episodes_list:
        tic = time.time()
        main(args, ep, pose_estimator)
        tac = time.time()
        prTimer(f"[Extract poses sapiens][{ep}]", tic, tac)
    
    return None


def gpu_worker(gpu_idx, episodes_list, args):
    """
    Worker function for multiprocessing that initializes a pose estimator 
    for a specific GPU and runs the processing loop.
    """
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(gpu_idx)
        device = torch.device("cuda", index=gpu_idx)
        
        prInfo(f"GPU {gpu_idx} worker starting with episodes: {episodes_list}")
        
        # Initialize pose estimator for this GPU (process-local)
        # Determine model path based on GPU capabilities and arguments
        if "A100" in torch.cuda.get_device_properties(gpu_idx).name and args.use_bfloat16:
            prInfo(f"GPU {gpu_idx}: Using A100 GPU with bfloat16")
            
            if args.num_keypoints == 308:
                if args.model == "03b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2")
                elif args.model == "06b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2")
                elif args.model == "1b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2")
                else:
                    raise NotImplementedError("Only 03b, 06b or 1b model are supported for now")
            else:
                raise NotImplementedError("Only 308 keypoints are supported for now")
        else:
            prInfo(f"GPU {gpu_idx}: Using GPU that is not an A100 or bfloat16 disabled")
            
            if args.num_keypoints == 308:
                if args.model == "03b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2")
                elif args.model == "06b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2")
                elif args.model == "1b":    
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
                else:
                    raise NotImplementedError("Only 03b, 06b or 1b model are supported for now")
            else:
                raise NotImplementedError("Only 308 keypoints are supported for now")
        
        # Build and configure the model
        use_torchscript = "_torchscript" in model_path
        pose_estimator = load_model(model_path, use_torchscript=use_torchscript)
        
        if not use_torchscript:
            # bfloat16 case, on A100
            dtype = torch.bfloat16
            pose_estimator.to(dtype)
            pose_estimator.to(device)
            pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
        else:
            pose_estimator = pose_estimator.half()
            pose_estimator = pose_estimator.to(device)
        
        prInfo(f"GPU {gpu_idx}: Pose estimator loaded and configured")
        
        # Run the processing loop
        processing_loop(args, episodes_list, pose_estimator)
        
        prSuccess(f"GPU {gpu_idx} worker completed all episodes")
        
    except Exception as e:
        prError(f"GPU {gpu_idx} worker failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, default="./output/rosbags_2025_03_26_15_25_25")
    parser.add_argument("--episodes", "-e", type=str, default="all")
    parser.add_argument("--model", "-m", type=str, default="03b")
    parser.add_argument("--batch_size", "-b", type=int, default=2)
    parser.add_argument("--num_keypoints", "-n", type=int, default=308)
    parser.add_argument("--sam2_masks", "-sam2", action="store_true", default=False)
    parser.add_argument("--use_bfloat16", "-b16", action="store_true", default=False)
    parser.add_argument("--ngpu", "-ngpu", type=int, default=0, help="Number of GPUs to use, default 0 means use all GPUs")

    args = parser.parse_args()
    
    assert args.num_keypoints in [17, 308, 133], "Number of keypoints should be 17, 308 or 133"
    assert args.model in ["03b", "06b","1b"], "Model should be 03b, 06b or 1b"
    assert os.path.isdir(args.dir), "Directory should exist"
    assert os.path.isdir(os.path.join(args.dir, "episodes")), "Episodes directory should exist"
    assert(torch.cuda.is_available()), "CUDA is not available"
    
    if args.episodes == "all":
        episodes_list = ["{:04d}".format(int(x)) for x in os.listdir(os.path.join(args.dir, "episodes"))]
    else:
        episodes_list = ["{:04d}".format(int(x)) for x in args.episodes.split(",")]

    episodes_list.sort()
    
    for ep in episodes_list:
        assert os.path.isdir(os.path.join(args.dir, "episodes", ep)), "Episode {} should exist".format(ep)

    # Count the number of images in each episode for load balancing
    images_per_episode = {}
    total_number_of_images = 0
    for ep in episodes_list:
        images_per_episode[ep] = len(os.listdir(os.path.join(args.dir, "episodes", ep, "images_360")))
        total_number_of_images += images_per_episode[ep]

    # Set the number of GPUs to use depending on the number of episodes and the number of GPUs available
    if args.ngpu == 0:
        args.ngpu = torch.cuda.device_count()
    args.ngpu = min(args.ngpu, len(episodes_list))
    prInfo("Using {} GPUs".format(args.ngpu))
        
    # Display GPU information (but don't initialize models in main process)
    for gpu_idx in range(args.ngpu):
        device_properties = torch.cuda.get_device_properties(gpu_idx)
        print("Will use GPU {} (idx = {}) | name : {} | memory : {:.2f} GB".format(torch.device("cuda", index = gpu_idx), gpu_idx, device_properties.name, device_properties.total_memory / 1024**3))
        
    prInfo("Processing episodes : {} (total number of images : {}) | Using {} GPUs".format(", ".join(episodes_list), total_number_of_images, args.ngpu))

    # Distribute episodes across GPUs to balance the load
    images_divided_per_ngpu = total_number_of_images // args.ngpu
    episodes_per_gpu = {idx: [] for idx in range(args.ngpu)} # dict with {gpu_idx: [episodes...], gpu_idx : [episodes...]}
    current_gpu_idx = 0
    current_gpu_images = 0
    for ep in episodes_list:
        images_in_episode = images_per_episode[ep]
        episodes_per_gpu[current_gpu_idx].append(ep)
        current_gpu_images += images_in_episode
        if current_gpu_images > images_divided_per_ngpu:
            # Move to next GPU to balance load
            current_gpu_idx += 1
            current_gpu_idx = min(current_gpu_idx, args.ngpu - 1) # Avoid going out of bounds
            current_gpu_images = 0
    
    # Display episode distribution
    for gpu_idx, episodes in episodes_per_gpu.items():
        gpu_images = sum(images_per_episode[ep] for ep in episodes)
        prInfo(f"GPU {gpu_idx}: {len(episodes)} episodes with {gpu_images} images total: {episodes}")

    # Use multiprocessing for multiple GPUs or single process for one GPU
    if args.ngpu == 1:
        # Single GPU - run directly without multiprocessing overhead
        prInfo("Single GPU mode - running directly")
        
        # Initialize model for single GPU processing
        # check if the gpu is a A100
        if "A100" in torch.cuda.get_device_properties(0).name and args.use_bfloat16:
            prInfo("Using A100 GPU")
            
            if args.num_keypoints == 308:
                if args.model == "03b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2")
                elif args.model == "06b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2")
                elif args.model == "1b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "bfloat16", "pose", "checkpoints", "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2")
                else:
                    raise NotImplementedError("Only 03b, 06b or 1b model are supported for now")
            else:
                raise NotImplementedError("Only 308 keypoints are supported for now")
        else:
            prInfo("Using GPU that is not an A100, cannot use bfloat16 anyway")
            
            if args.num_keypoints == 308:
                if args.model == "03b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2")
                elif args.model == "06b":
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2")
                elif args.model == "1b":    
                    model_path = os.path.join(here, "..", "checkpoints", "sapiens", "torchscript", "pose", "checkpoints", "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
                else:
                    raise NotImplementedError("Only 03b, 06b or 1b model are supported for now")
            else:
                raise NotImplementedError("Only 308 keypoints are supported for now")

        ### Build the model from a checkpoint file ###
        use_torchscript = "_torchscript" in model_path
        pose_estimator = load_model(model_path, use_torchscript=use_torchscript)
        
        if not use_torchscript:
            # bfloat16 case, on A100
            dtype = torch.bfloat16
            pose_estimator.to(dtype)
            pose_estimator = torch.compile(pose_estimator, mode="max-autotune", fullgraph=True)
        else:
            pose_estimator = pose_estimator.half()
            pose_estimator = pose_estimator.to(torch.device("cuda"))
        
        processing_loop(args, episodes_per_gpu[0], pose_estimator)
        prSuccess("Single GPU processing completed!")
    else:
        # Multiple GPUs - use multiprocessing
        prInfo("Multi-GPU mode - using torch multiprocessing")
        
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # Create and start worker processes
        processes = []
        for gpu_idx in range(args.ngpu):
            if len(episodes_per_gpu[gpu_idx]) > 0:  # Only start process if there are episodes to process
                p = mp.Process(target=gpu_worker, args=(gpu_idx, episodes_per_gpu[gpu_idx], args))
                p.start()
                processes.append(p)
                prInfo(f"Started process for GPU {gpu_idx}")
        
        # Wait for all processes to complete
        prInfo("Waiting for all GPU processes to complete...")
        for i, p in enumerate(processes):
            p.join()
            if p.exitcode == 0:
                prSuccess(f"Process {i} completed successfully")
            else:
                prError(f"Process {i} failed with exit code {p.exitcode}")
        
        prSuccess("All GPU processes completed!")
    