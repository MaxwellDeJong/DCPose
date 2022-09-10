#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import sys

sys.path.insert(0, os.path.abspath('../'))

from PIL import Image
import copy
import cv2
import logging
import json
import numpy as np
from typing import List, Optional

from datasets.process.keypoints_ord import coco2posetrack_ord_infer
from engine.core.vis_helper import add_poseTrack_joint_connection_to_image, add_bbox_in_image
from object_detector.detector_yolov3 import load_eval_model
from object_detector.detector_yolov3 import inference_yolov3_from_img
from object_detector.models import Darknet
from tools.inference import get_inference_model
from tools.inference import get_inference_preprocessing_transforms
from tools.inference import inference_PE


_ZERO_FILL = 8
logger = logging.getLogger(__name__)

class Video:
  """Class representing a video for pose estimation."""

  def __init__(self, video_path: str, output_dir: str, frame_dir: Optional[str]=None):
    self._path = video_path
    self._output_dir = output_dir
    if frame_dir:
      self._frame_dir = frame_dir
    else:
      self._frame_dir = os.path.join(os.path.dirname(self._path), 'frames')
    self._frames: List[VideoFrame] = []

  @property
  def basename(self):
    """Return base name of video filename."""
    full_video_name = os.path.basename(self._path)
    # Remove extension
    return full_video_name.split(".")[0]

  @property
  def length(self):
    """Return number of frames in the video."""
    return len(self._frames)

  def split_to_frames(self):
    """Split a video into individual frames that are saved."""
    image_save_path = os.path.join(self._frame_dir, self.basename)
    cap = cv2.VideoCapture(self._path)
    isOpened: bool = cap.isOpened()
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not os.path.exists(image_save_path):
      os.makedirs(image_save_path)
    assert isOpened, "Can't find video"
    for index in range(video_length):
      (flag, frame_data) = cap.read()
      frame_data = cv2.transpose(frame_data)
      frame_name = "{}.jpg".format(str(index).zfill(_ZERO_FILL))
      frame_path = os.path.join(image_save_path, frame_name)
      if flag:
        cv2.imwrite(frame_path, frame_data, [cv2.IMWRITE_JPEG_QUALITY, 100])
        self._frames.append(VideoFrame(frame_path))

  def estimate_pose(self, image_transforms, pose_estimation_model):
    """Estimate person pose across the video frames."""
    for idx, frame in enumerate(self._frames):
      prev_idx = max(idx - 1, 0)
      prev_frame = self._frames[prev_idx]
      next_idx = min(idx + 1, self.length - 1)
      next_frame = self._frames[next_idx]
      for bbox in frame.bboxes:
        raw_keypoints = inference_PE(frame.path, prev_frame.path, next_frame.path, bbox, image_transforms, pose_estimation_model)
        frame._keypoints.append(raw_keypoints)
    self._done_pose_estimation = True

  def export_frame_detections(
      self, detection_model: Darknet, apply_backfill: bool=True):
    """Save all frames with the person detection overlayed."""
    detection_dir = os.path.join(self._output_dir, 'detections/')
    prev_bbox = []
    backfilled = False
    for frame in self._frames:
      frame.detect_person(detection_model)
      if apply_backfill:
        if not frame._bboxes:
          frame._bboxes = prev_bbox
          backfilled = True
          logging.info(
            'No detection found in frame. Applying detection backfill.')
        else:
          prev_bbox = frame._bboxes
          backfilled = False
      frame.draw_detection(backfilled_detection=backfilled)
      frame.export_detection_frame(detection_dir)

  def export_frame_pose_estimations(self):
    """Save all frames with the pose estimation overlayed."""
    pose_estimation_dir = os.path.join(self._output_dir, 'pose_estimation/')
    for frame in self._frames:
      frame.draw_pose_estimation()
      frame.export_pose_estimation_frame(pose_estimation_dir)

  def export_frame_json(self):
    """Save metadata created during pose estimation."""
    json_dir = os.path.join(self._output_dir, 'json/')
    for frame in self._frames:
      frame.export_json(json_dir)

  def _export_video(
      self, images: List[np.ndarray], video_name: str, fps: int):
    """Export a list of images as a video."""
    size = (images[0].shape[1], images[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    video = cv2.VideoWriter(video_name, fourcc, fps, size)
    for image in images:
      video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

  def export_detection_video(self, fps: int=25):
    """Export a video visualizing the person detection."""
    detection_images = [frame.detection_image for frame in self._frames]
    video_name = os.path.join(self._output_dir, self.basename + '_detection.mp4')
    self._export_video(detection_images, video_name, fps=fps)

  def export_pose_estimation_video(self, fps: int=25):
    """Export a video visualizing the pose estimation."""
    pose_estimation_images = [
      frame.pose_estimation_image for frame in self._frames]
    video_name = os.path.join(self._output_dir,
                              self.basename + '_pose_estimation.mp4')
    self._export_video(pose_estimation_images, video_name, fps=fps)

  def perform_detection(self, detection_model: Darknet):
    """Perform person detection and export resulting video."""
    self.export_frame_detections(detection_model)
    self.export_detection_video()

  def perform_pose_estimation(
      self, pose_estimation_preprocessing_transforms, pose_estimation_model):
    """Perform pose estimation and export resulting video."""
    self.estimate_pose(
      pose_estimation_preprocessing_transforms, pose_estimation_model)
    self.export_frame_pose_estimations()
    self.export_pose_estimation_video()

  def infer_pose(self,
                 detection_model: Darknet,
                 pose_estimation_preprocessing_transforms,
                 pose_estimation_model):
    """Complete end-to-end pose estimation."""
    self.split_to_frames()
    self.perform_detection(detection_model)
    self.perform_pose_estimation(
      pose_estimation_preprocessing_transforms, pose_estimation_model)
    self.export_frame_json()


class VideoFrame:
  """Class to represent a single frame of a video."""

  def __init__(self, image_path: str):
    self._path = image_path
    self._image: np.ndarray = np.asarray(Image.open(self._path))
    self._detection_image: Optional[np.ndarray] = None
    self._pose_estimation_image: Optional[np.ndarray] = None
    self._bboxes: List = []
    self._keypoints: List = []

  @property
  def path(self) -> str:
    """Path to the saved image."""
    return self._path

  @property
  def basename(self):
    """Return base name of frame filename."""
    full_frame_name = os.path.basename(self._path)
    # Remove extension
    return full_frame_name.split(".")[0]

  @property
  def bboxes(self) -> List:
    """Bounding boxes associated with frame."""
    return self._bboxes

  @property
  def keypoints(self) -> List:
    """Keypoints associated with frame."""
    return self._keypoints

  @property
  def image(self) -> np.ndarray:
    """Image data as numpy array."""
    return self._image

  @property
  def detection_image(self) -> np.ndarray:
    """Image data with detection overlayed."""
    return self._detection_image

  @property
  def pose_estimation_image(self) -> np.ndarray:
    """Image data with pose estimation overlayed."""
    return self._pose_estimation_image

  def detect_person(self, model: Darknet):
    """Perform person detection on frame."""
    self._bboxes = inference_yolov3_from_img(self.image, model)

  def draw_detection(self, backfilled_detection: bool=False):
    """Draw the result of performing person detection."""
    self._detection_image = copy.deepcopy(self.image)
    if backfilled_detection:
      color = (255, 100, 100)
    else:
      color = (255, 255, 255)
    for bbox in self.bboxes:
      xyxy_box = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
      self._detection_image = add_bbox_in_image(
        self.detection_image, xyxy_box, color=color)

  def draw_pose_estimation(self):
    """Draw the result of performing pose estimation."""
    self._pose_estimation_image = copy.deepcopy(self.image)
    for keypoints in self.keypoints:
      new_coord = coco2posetrack_ord_infer(keypoints[0])
      self._pose_estimation_image = add_poseTrack_joint_connection_to_image(
        self._pose_estimation_image,
        new_coord,
        sure_threshold=0.3,
        flag_only_draw_sure=True)

  def export_pose_estimation_frame(self, pose_estimation_output_dir: str):
    """Export an image with the pose estimation visualized."""
    if not os.path.exists(pose_estimation_output_dir):
      os.makedirs(pose_estimation_output_dir)
    pose_estimation_frame_fname = os.path.join(pose_estimation_output_dir,
                                               self.basename + '.jpg')
    cv2.imwrite(pose_estimation_frame_fname,
                cv2.cvtColor(self._pose_estimation_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 100])

  def export_detection_frame(self, detection_output_dir: str):
    """Export an image with the person detection visualized."""
    if not os.path.exists(detection_output_dir):
      os.makedirs(detection_output_dir)
    detection_frame_fname = os.path.join(
      detection_output_dir, self.basename + '.jpg')
    cv2.imwrite(detection_frame_fname,
                cv2.cvtColor(self._detection_image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 100])

  def export_json(self, json_output_dir: str):
    """Export JSON data containing bounding boxes and keypoints."""
    if not os.path.exists(json_output_dir):
      os.makedirs(json_output_dir)
    joints_info = {'frame_name': self.path,
                   'frame_bboxes': self.bboxes,
                   'frame_keypoints': 
                     [keypoints.tolist()[0] for keypoints in self.keypoints]}
    json_frame_fname = os.path.join(json_output_dir, self.basename + '.json')
    with open(json_frame_fname, 'w') as json_file:
      json.dump(joints_info, json_file)


if __name__ == '__main__':
  video_path = './input/squat_2.mp4'
  output_dir = './outputs_squat2/'
  frame_dir = None
  v = Video(video_path, output_dir)

  detection_model = load_eval_model()
  pose_estimation_preprocessing_transforms = (
    get_inference_preprocessing_transforms())
  pose_estimation_model = get_inference_model()

  v.infer_pose(detection_model,
               pose_estimation_preprocessing_transforms,
               pose_estimation_model)
