from object_detector import models
from object_detector import detector_utils

import PIL
import argparse
import numpy as np
import os
import torch


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--config_path",
                    type=str,
                    default="object_detector/config/yolov3.cfg",
                    help="path to model config file")
parser.add_argument("--weights_path",
                    type=str,
                    default="DcPose_supp_files/object_detector/YOLOv3/yolov3.weights",
                    help="path to weights file")
parser.add_argument("--conf_thres",
                    type=float,
                    default=0.2,
                    help="object confidence threshold")
parser.add_argument("--nms_thres",
                    type=float,
                    default=0.4,
                    help="iou threshold for non-maximum suppression")
parser.add_argument("--batch_size",
                    type=int,
                    default=1,
                    help="size of the batches")
parser.add_argument("--n_cpu",
                    type=int,
                    default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size",
                    type=int,
                    default=416,
                    help="size of each image dimension")
parser.add_argument("--checkpoint_model",
                    type=str,
                    help="path to checkpoint model")
opt = parser.parse_args()
###
this_file_path = __file__
tracking_network_path = '/home/max/Desktop/pose_estimation/DCPose/'
opt.config_path = os.path.join(tracking_network_path, opt.config_path)
opt.weights_path = os.path.join(tracking_network_path, opt.weights_path)
print("Detector YOLOv3 options:", opt)

_Tensor_Type = (
  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)


def load_eval_model():
  """Load a yolov3 detector model for evaluation."""
  cuda = torch.cuda.is_available()
  # Set up model
  model = models.Darknet(opt.config_path, img_size=opt.img_size)
  if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
  else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))
  if cuda:
    model.cuda()
  model.eval()  # Set in evaluation mode
  return model


def inference_yolov3_from_img(img: np.ndarray, model: models.Darknet):
  input_img = detector_utils.preprocess_img_for_yolo(img)
  # Configure input
  input_img = torch.autograd.Variable(input_img.type(_Tensor_Type))

  # Get detections
  with torch.no_grad():
    detections = model(input_img)
    detections = detector_utils.non_max_suppression(detections,
                                                    opt.conf_thres,
                                                    opt.nms_thres)[0]
    if detections is None:
      return []
    else:
      detections = detections.data.cpu().numpy()

  # The amount of padding that was added
  pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
  pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
  # Image height and width after padding is removed
  unpad_h = opt.img_size - pad_y
  unpad_w = opt.img_size - pad_x

  # Draw bounding boxes and labels of detections
  human_candidates = []
  if detections is not None:
    for x1, y1, x2, y2, cls_conf, cls_pred in detections:
      # Rescale coordinates to original dimensions
      box_h = ((y2 - y1) / unpad_h) * img.shape[0]
      box_w = ((x2 - x1) / unpad_w) * img.shape[1]
      y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
      x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

      if int(cls_pred) == 0:
        human_candidate = [x1, y1, box_w, box_h]
        human_candidates.append(human_candidate)
  return human_candidates


if __name__ == "__main__":
  img_path = "/export/guanghan/PyTorch-YOLOv3/data/samples/messi.jpg"
  img = np.array(PIL.Image.open(img_path))
  human_candidates = inference_yolov3_from_img(img)
  print("human_candidates:", human_candidates)
