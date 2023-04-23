
import numpy as np
import matplotlib.pyplot as plt
from scripts.ld_convertor import df2rgba

from pytoshop import layers
import pytoshop

from psd_tools import PSDImage 
from psd_tools.psd.layer_and_mask import LayerRecord
from psd_tools.constants import BlendMode
from PIL import Image

import random, string
import os

def randomname(n):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)


def img_plot(df):
  img = df2rgba(df).astype(np.uint8)
  plt.imshow(img)
  plt.show()


def add_psd(psd, img, name, mode):

  layer_1 = layers.ChannelImageData(image=img[:, :, 3], compression=1)
  layer0 = layers.ChannelImageData(image=img[:, :, 0], compression=1) 
  layer1 = layers.ChannelImageData(image=img[:, :, 1], compression=1) 
  layer2 = layers.ChannelImageData(image=img[:, :, 2], compression=1) 

  new_layer = layers.LayerRecord(channels={-1: layer_1, 0: layer0, 1: layer1, 2: layer2}, 
                                  top=0, bottom=img.shape[0], left=0, right=img.shape[1], 
                                  blend_mode=mode,
                                  name=name, 
                                  opacity=255, 
                                  )
  psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
  return psd

def save_psd(input_image, layers, names, modes):
  psd = pytoshop.core.PsdFile(num_channels=3, height=input_image.shape[0], width=input_image.shape[1])

  for idx, img_list in enumerate(layers):
      for num, output in enumerate(img_list):
          psd = add_psd(psd, output, names[idx] + str(num), modes[idx])
          

  path = os.getcwd()
  name = randomname(10)

  with open(f"{path}/output/output_{name}.psd", 'wb') as fd2:
      psd.write(fd2)  

  return f"{path}/output/output_{name}.psd"

