from __future__ import print_function
from __future__ import division

import numpy as np
import rawpy, cv2

USE_OPENCV = False

with rawpy.imread('image.dng') as raw:
  # print(raw.raw_image.shape) # (5472, 7296)
  print(raw.color_desc)
  print(type(raw.raw_image))
  h, w = raw.raw_image.shape
  if USE_OPENCV:
    image = cv2.cvtColor(raw.raw_image.astype(np.uint8), cv2.COLOR_BayerRG2BGR)
  else:
    image = raw.postprocess()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  print(image.shape)
  image = cv2.resize(image, (int(w * 0.1), int(h * 0.1)))
  cv2.imshow('Demo', image)
  cv2.waitKey()
