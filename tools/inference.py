import argparse
import cv2
import logging
import numpy as np
import os
import torch

from datasets.process import get_affine_transform
from datasets.process import get_final_preds
from datasets.transforms import build_transforms
from pose_estimation.config import get_cfg, update_config
from pose_estimation.zoo import build_model
from utils.common import INFERENCE_PHASE
from utils.utils_bbox import box2cs
from utils.utils_image import read_image, save_image


_IMAGE_SIZE = np.array([288, 384])
_ASPECT_RATIO = _IMAGE_SIZE[0] * 1.0 / _IMAGE_SIZE[1]


def parse_args():
  parser = argparse.ArgumentParser(description='Inference pose estimation Network')

  parser.add_argument('--cfg', help='experiment configure file name', required=False, type=str,
            default="./configs/posetimation/DcPose/posetrack17/model_RSN_inference.yaml")
  parser.add_argument('--PE_Name', help='pose estimation model name', required=False, type=str,
            default='DcPose')
  parser.add_argument('-weight', help='model weight file', required=False, type=str
            , default='./DcPose_supp_files/pretrained_models/DCPose/PoseTrack17_DCPose.pth')
  parser.add_argument('--gpu_id', default='0')
  parser.add_argument('opts', help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)

  # philly
  args = parser.parse_args()
  args.rootDir = os.path.abspath('../')
  args.cfg = os.path.abspath(os.path.join(args.rootDir, args.cfg))
  args.weight = os.path.abspath(os.path.join(args.rootDir, args.weight))
  return args


def get_inference_model():
  logger = logging.getLogger(__name__)
  args = parse_args()
  cfg = get_cfg(args)
  update_config(cfg, args)
  logger.info("load :{}".format(args.weight))

  model = build_model(cfg, INFERENCE_PHASE)
  if torch.cuda.is_available():
    model = model.cuda()

  checkpoint_dict = torch.load(args.weight)
  model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
  model.load_state_dict(model_state_dict)
  return model


def get_inference_preprocessing_transforms():
  return build_transforms(None, INFERENCE_PHASE)



def image_preprocess(image_path: str, prev_image: str, next_image: str, center, scale, image_transforms):
  trans_matrix = get_affine_transform(center, scale, 0, _IMAGE_SIZE)
  image_data = read_image(image_path)
  image_data = cv2.warpAffine(image_data, trans_matrix, (int(_IMAGE_SIZE[0]), int(_IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR)
  image_data = image_transforms(image_data)
  if prev_image is None or next_image is None:
    return image_data
  else:
    prev_image_data = read_image(prev_image)
    prev_image_data = cv2.warpAffine(prev_image_data, trans_matrix, (int(_IMAGE_SIZE[0]), int(_IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR)
    prev_image_data = image_transforms(prev_image_data)

    next_image_data = read_image(next_image)
    next_image_data = cv2.warpAffine(next_image_data, trans_matrix, (int(_IMAGE_SIZE[0]), int(_IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR)
    next_image_data = image_transforms(next_image_data)

    return image_data, prev_image_data, next_image_data


def inference_PE(input_image: str, prev_image: str, next_image: str, bbox, image_transforms, model):
  """
    input_image : input image path
    prev_image : prev image path
    next_image : next image path
    inference pose estimation
  """
  center, scale = box2cs(bbox, _ASPECT_RATIO)
  target_image_data, prev_image_data, next_image_data = image_preprocess(input_image, prev_image, next_image, center, scale, image_transforms)

  target_image_data = target_image_data.unsqueeze(0)
  prev_image_data = prev_image_data.unsqueeze(0)
  next_image_data = next_image_data.unsqueeze(0)

  concat_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
  margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
  model.eval()

  predictions = model(concat_input, margin=margin)

  pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), [center], [scale])
  pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)

  return pred_keypoints


def inference_PE_batch(input_image_list: list, prev_image_list: list, next_image_list: list, bbox_list: list, image_transforms, model):
  """
    input_image : input image path
    prev_image : prev image path
    next_image : next image path
    inference pose estimation
  """
  batch_size = len(input_image_list)

  batch_input = []
  batch_margin = []
  batch_center = []
  batch_scale = []
  for batch_index in range(batch_size):
    bbox = bbox_list[batch_index]
    input_image = input_image_list[batch_index]
    prev_image = prev_image_list[batch_index]
    next_image = next_image_list[batch_index]

    center, scale = box2cs(bbox, _ASPECT_RATIO)
    batch_center.append(center)
    batch_scale.append(scale)

    target_image_data, prev_image_data, next_image_data = image_preprocess(input_image, prev_image, next_image, center, scale, image_transforms)

    target_image_data = target_image_data.unsqueeze(0)
    prev_image_data = prev_image_data.unsqueeze(0)
    next_image_data = next_image_data.unsqueeze(0)

    one_sample_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
    margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()

    batch_input.append(one_sample_input)
    batch_margin.append(margin)
  batch_input = torch.cat(batch_input, dim=0).cuda()
  batch_margin = torch.cat(batch_margin, dim=0).cuda()
  # concat_input = torch.cat((target_image_data, prev_image_data, next_image_data), 1).cuda()
  # margin = torch.stack([torch.tensor(1).unsqueeze(0), torch.tensor(1).unsqueeze(0)], dim=1).cuda()
  model.eval()

  predictions = model(batch_input, margin=batch_margin)

  pred_joint, pred_conf = get_final_preds(predictions.cpu().detach().numpy(), batch_center, batch_scale)
  pred_keypoints = np.concatenate([pred_joint.astype(int), pred_conf], axis=2)

  return pred_keypoints
