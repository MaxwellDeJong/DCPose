#!/usr/bin/python
# -*- coding:utf8 -*-
import cv2
import numpy as np
import os
from typing import Optional

from .utils_folder import create_folder, folder_exists, list_immediate_childfile_paths


def video2images(video_path: str,
                 outimages_path: Optional[str]=None,
                 zero_fill: int=8):
  """Save video frames as individual images."""
  cap = cv2.VideoCapture(video_path)
  isOpened = cap.isOpened()
  video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  if outimages_path is not None:
    if not os.path.exists(outimages_path):
      os.makedirs(outimages_path)
  assert isOpened, "Can't find video"
  for index in range(video_length):
    (flag, data) = cap.read()
    file_name = "{}.jpg".format(str(index).zfill(zero_fill))  # start from zero
    if outimages_path is not None:
      file_path = os.path.join(outimages_path, file_name)
    else:
      create_folder("output")
      file_path = os.path.join("output", file_name)
    if flag:
      cv2.imwrite(file_path, data, [cv2.IMWRITE_JPEG_QUALITY, 100])


def image2video(image_dir: str, name: str, fps: int=25):
  """Stitch together images to form a video."""
  image_path_list = []
  for image_path in list_immediate_childfile_paths(image_dir):
    image_path_list.append(image_path)
  image_path_list.sort()
  temp = cv2.imread(image_path_list[0])
  size = (temp.shape[1], temp.shape[0])
  fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
  video = cv2.VideoWriter('./output/' + name + '.mp4', fourcc, fps, size)
  for image_path in image_path_list:
    if image_path.endswith(".jpg"):
      image_data_temp = cv2.imread(image_path)
      video.write(image_data_temp)
  print("Video done！")
