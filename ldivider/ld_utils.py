
import numpy as np
import matplotlib.pyplot as plt
from ldivider.ld_convertor import df2rgba

from pytoshop import layers
from pytoshop.user import nested_layers
import pytoshop

from PIL import Image

import random, string
import os

import psd_tools
from psd_tools.psd import PSD

import requests
from tqdm import tqdm


import pickle
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
  #gp = nested_layers.Group()
  #gp.layers = [new_layer]
  psd.layer_and_mask_info.layer_info.layer_records.append(new_layer)
  return psd

def load_seg_model(model_dir):
  folder = model_dir
  file_name = 'sam_vit_h_4b8939.pth'
  url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

  file_path = os.path.join(folder, file_name)
  if not os.path.exists(file_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    with open(file_path, 'wb') as f, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)



def load_masks(output_dir):
  pkl_path = os.path.join(output_dir, "tmp", "seg_layer", "sorted_masks.pkl")
  with open(pkl_path, 'rb') as f:
    masks = pickle.load(f)
  return masks

def save_psd(input_image, layers, names, modes, output_dir, layer_mode):
  psd = pytoshop.core.PsdFile(num_channels=3, height=input_image.shape[0], width=input_image.shape[1])
  if layer_mode == "normal":
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
  else:
    for idx, output in enumerate(layers[0]):
      psd = add_psd(psd, layers[0][idx], names[0] + str(idx), modes[0])
      psd = add_psd(psd, layers[1][idx], names[1] + str(idx), modes[1])
      psd = add_psd(psd, layers[2][idx], names[2] + str(idx), modes[2])
      psd = add_psd(psd, layers[3][idx], names[3] + str(idx), modes[3])
      psd = add_psd(psd, layers[4][idx], names[4] + str(idx), modes[4])

  name = randomname(10)

  with open(f"{output_dir}/output_{name}.psd", 'wb') as fd2:
      psd.write(fd2)

  return f"{output_dir}/output_{name}.psd"

def divide_folder(psd_path, input_dir, mode):
  with open(f'{input_dir}/empty.psd', "rb") as fd:
    psd_base = PSD.read(fd)
  with open(psd_path, "rb") as fd:
    psd_image = PSD.read(fd)

  if mode == "normal":
     add_num = 3
  else:
     add_num = 5

  base_records_list = list(psd_base.layer_and_mask_information.layer_info.layer_records)
  image_records_list = list(psd_image.layer_and_mask_information.layer_info.layer_records)

  merge_list = []
  for idx, record in enumerate(image_records_list):
      if idx % add_num == 0:
          merge_list.append(base_records_list[0])
      merge_list.append(record)
      if idx % add_num == (add_num - 1):
          merge_list.append(base_records_list[2])

  psd_image.layer_and_mask_information.layer_info.layer_records = psd_tools.psd.layer_and_mask.LayerRecords(merge_list)
  psd_image.layer_and_mask_information.layer_info.layer_count = len(psd_image.layer_and_mask_information.layer_info.layer_records)

  folder_channel = psd_base.layer_and_mask_information.layer_info.channel_image_data[0]
  image_channel = psd_image.layer_and_mask_information.layer_info.channel_image_data

  channel_list = []
  for idx, channel in enumerate(image_channel):
      if idx % add_num == 0:
          channel_list.append(folder_channel)
      channel_list.append(channel)
      if idx % add_num == (add_num - 1):
          channel_list.append(folder_channel)

  psd_image.layer_and_mask_information.layer_info.channel_image_data =  psd_tools.psd.layer_and_mask.ChannelImageData(channel_list)
  with open(psd_path, 'wb') as fd:
      psd_image.write(fd)

  return psd_path
