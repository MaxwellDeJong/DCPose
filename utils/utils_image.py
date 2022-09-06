#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import cv2
from .utils_folder import create_folder


def read_image(image_path):
  if not os.path.exists(image_path):
    raise Exception("Failed to read image from path : {}".format(image_path))
  img = cv2.imread(image_path)

  return img


def save_image(image_save_path, image_data):
  create_folder(os.path.dirname(image_save_path))
  return cv2.imwrite(image_save_path, image_data, [100])
