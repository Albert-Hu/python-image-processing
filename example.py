from __future__ import print_function
from __future__ import division

import argparse, rawpy
import cv2 as cv
import numpy as np

def noisy(image, shape):
  mean = 0
  sigma = .1
  normalized_image = image / 255.
  noise = np.random.normal(mean, sigma, image.shape)
  noise_image = np.clip(normalized_image + noise, 0, 1) * 255
  return noise_image.astype(np.uint8)

def process(image_path):
  with rawpy.imread(image_path) as raw:
    image = raw.postprocess()
    h, w, c = image.shape
    ratio = max((800.0 / w), (600.0 / h))
    h = int(h * ratio)
    w = int(w * ratio)
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    image = cv.resize(image, (w, h))
    is_opened = lambda window: cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) > 0
    noise_image = noisy(image, (h, w, c))
    denoised_image = cv.GaussianBlur(noise_image, (3, 3), 0)
    cv.imshow('original', image)
    cv.imshow('noise', noise_image)
    cv.imshow('denoise', denoised_image)
    while is_opened('original') and is_opened('noise') and is_opened('denoise'):
      if (cv.waitKey(300) & 0xff) in [ord('q'), ord('Q')]:
        break
    cv.destroyAllWindows()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('image', type=str)
  args = parser.parse_args()
  process(args.image)
