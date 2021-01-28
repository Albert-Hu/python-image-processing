from __future__ import print_function
from __future__ import division

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float, img_as_ubyte
from skimage.util import random_noise

import argparse, rawpy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def process(dev_num):
  camera = cv.VideoCapture(dev_num)
  camera.set(cv.CAP_PROP_FRAME_WIDTH, 640)
  camera.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
  ret, image = camera.read()
  while ret:
    if (cv.waitKey(10) & 0xFF) in [ord('q'), ord('Q'), 27]:
      break
    original = img_as_float(image)
    sigma = 0.155
    noisy = random_noise(original, var=sigma**2)
    cv.imshow('Webcam', np.concatenate((image, img_as_ubyte(noisy)), axis=1))
    ret, image = camera.read()
  cv.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('dev_num', type=int)
  args = parser.parse_args()
  process(args.dev_num)
