import cv2
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import color 
from convertor import skimage_rgb2lab, df2rgba, rgba2df, hsv2df
from utils import img_plot
from bg_remover import get_foreground

def calc_ciede(mean_list, cls_list):
  cls_no = []
  tgt_no = []
  ciede_list = []
  for i in range(len(mean_list)):
    img_1 = np.array(mean_list[i][:3])
    for j in range(len(mean_list)):
      if i == j:
        continue
      img_2 = np.array(mean_list[j][:3])
      ciede = color.deltaE_ciede2000(skimage_rgb2lab(img_1), skimage_rgb2lab(img_2))[0][0]
      ciede_list.append(ciede)
      cls_no.append(cls_list[i])
      tgt_no.append(cls_list[j])
  ciede_df = pd.DataFrame({"cls_no": cls_no, "tgt_no": tgt_no, "ciede2000": ciede_list})
  return ciede_df

def get_mask(df, cls_no):
  mask = df.copy()
  mask.loc[df["label"] != cls_no, ["r","g","b"]] = 0
  mask.loc[df["label"] == cls_no, ["r","g","b"]] = 255
  mask = cv2.cvtColor(df2rgba(mask).astype(np.uint8), cv2.COLOR_RGBA2GRAY)
  return mask

def fill_mean_color(img_df, mask):
  df_img = df2rgba(img_df).astype(np.uint8)
  if len(df_img.shape) == 3:
      mask = np.repeat(mask[:, :, np.newaxis], df_img.shape[-1], axis=-1)
  masked_img = np.where(mask == 0, 0, df_img)
  mean = np.mean(masked_img[mask != 0].reshape(-1, df_img.shape[-1]), axis=0)

  img_df["r"] = mean[0]
  img_df["g"] = mean[1]
  img_df["b"] = mean[2]
  
  return img_df, mean

def get_blur_cls(img, cls, size):
  blur_img = cv2.blur(img, (size, size))
  blur_df = rgba2df(blur_img)
  blur_df["label"] = cls
  img_list = []
  mean_list = []
  cls_list = list(cls.unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask(blur_df, cls_no)
    img_df = blur_df.copy()
    img_df.loc[blur_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
    mean_list.append(mean)
  return img_list, mean_list, cls_list

def get_cls_update(ciede_df, df, threshold):
    set_list = [frozenset({cls, tgt}) for cls, tgt in ciede_df[ciede_df['ciede2000'] < threshold][['cls_no', 'tgt_no']].to_numpy()]
    merge_set = []
    while set_list:
        set_a = set_list.pop()
        merged = False
        for i, set_b in enumerate(merge_set):
            if set_a & set_b:
                merge_set[i] |= set_a
                merged = True
                break
        if not merged:
            merge_set.append(set_a)
    merge_dict = {}
    for merge in merge_set:
        cls_counts = {cls: len(df[df['label'] == cls]) for cls in merge}
        max_cls = max(cls_counts, key=cls_counts.get)
        for cls in merge:
            merge_dict[cls] = max_cls
    return merge_dict


def get_color_dict(mean_list, cls_list):
  color_dict = {}
  for idx, mean in enumerate(mean_list):
    color_dict.update({cls_list[idx]:{"r":mean[0],"g":mean[1],"b":mean[2], }})
  return color_dict

def get_update_df(df, merge_dict, mean_list, cls_list):
  update_df = df.copy()
  update_df["label"] = update_df["label"].apply(lambda x: x if x not in merge_dict.keys() else merge_dict[x])
  color_dict = get_color_dict(mean_list, cls_list)
  update_df["r"] = update_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  update_df["g"] = update_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  update_df["b"] = update_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)    
  return update_df, color_dict

def split_img_df(df, show=False):
  img_list = []
  for cls_no in tqdm(list(df["label"].unique())):
    img_df = df.copy()
    img_df.loc[df["label"] != cls_no, ["a"]] = 0 
    df_img = df2rgba(img_df).astype(np.uint8)
    img_list.append(df_img)
  return img_list


<<<<<<< HEAD
def get_base(img, roop, cls_num, threshold, size, bg_split = True, debug=False):
  #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
  if bg_split == False:
    df = rgba2df(img)
    df_list = [df]
  else:
    h_split = 256
    v_split = 256
    n_cluster = 500
    alpha = 50
    th_rate = 0.1
    df_list = get_foreground(img, h_split, v_split, n_cluster, alpha, th_rate)

  output_list = []
  print(f"df_list:{len(df_list)}")
  
  for idx, df in enumerate(df_list):
    output_df = df.copy()
    cls = MiniBatchKMeans(n_clusters = cls_num)
    cls.fit(df[["r","g","b"]])
    df["label"] = cls.labels_ 
    df["label"] = df["label"].astype(str) + f"_{idx}"
    for i in range(roop):
      if i !=0:
        img = df2rgba(df).astype(np.uint8)
      blur_list, mean_list, cls_list = get_blur_cls(img, df["label"], size)
      ciede_df = calc_ciede(mean_list, cls_list)
      merge_dict = get_cls_update(ciede_df, df, threshold)
      update_df, color_dict = get_update_df(df, merge_dict, mean_list, cls_list)
      df = update_df
      if debug==True:
        img_plot(df)
    output_df["label"] = df["label"]
    output_list.append(output_df)

  output_df = pd.concat(output_list).sort_index()

=======
def get_base(img, loops, cls_num, threshold, size, debug=False):
  #img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
  df = rgb2df(img)
  output_df = df.copy()
  cls = KMeans(n_clusters = cls_num)
  cls.fit(df[["r","g","b"]])
  df["label"] = cls.labels_
  for i in range(loops):
    if i !=0:
      img = df2rgba(df).astype(np.uint8)
    blur_list, mean_list, cls_list = get_blur_cls(img, df["label"], size)
    ciede_df = calc_ciede(mean_list, cls_list)
    merge_dict = get_cls_update(ciede_df, df, threshold)
    update_df, color_dict = get_update_df(df, merge_dict, mean_list, cls_list)
    df = update_df
    if debug==True:
      img_plot(df)

  output_df["label"] = df["label"]
>>>>>>> ed80020c3e61f3455b487f13c9dab365e25285c6
  mean_list = []
  cls_list = list(output_df["label"].unique())
  for cls_no in tqdm(cls_list):
    mask = get_mask(output_df, cls_no)
    img_df = output_df.copy()
    img_df.loc[output_df["label"] != cls_no, ["a"]] = 0 
    img_df, mean = fill_mean_color(img_df, mask)
    mean_list.append(mean)

  color_dict = get_color_dict(mean_list, cls_list)
  output_df["r"] = output_df.apply(lambda x: color_dict[x["label"]]["r"], axis=1)
  output_df["g"] = output_df.apply(lambda x: color_dict[x["label"]]["g"], axis=1)
  output_df["b"] = output_df.apply(lambda x: color_dict[x["label"]]["b"], axis=1)  
  
  return output_df


def get_normal_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)
  hsv_df = hsv2df(cv2.cvtColor(df2rgba(df).astype(np.uint8), cv2.COLOR_RGB2HSV))
  hsv_org = hsv2df(cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV))

  hsv_org["bright_flg"] = hsv_df["v"] < hsv_org["v"]
  bright_df = org_df.copy()
  bright_df["bright_flg"] = hsv_org["bright_flg"]
  bright_df["a"] = bright_df.apply(lambda x: 255 if x["bright_flg"] == True else 0, axis=1)
  bright_df["label"] = df["label"]
  bright_layer_list = split_img_df(bright_df, show=False)

  hsv_org["shadow_flg"] = hsv_df["v"] >= hsv_org["v"]
  shadow_df = rgba2df(input_image)
  shadow_df["shadow_flg"] = hsv_org["shadow_flg"]
  shadow_df["a"] = shadow_df.apply(lambda x: 255 if x["shadow_flg"] == True else 0, axis=1)
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)
  
  return base_layer_list, bright_layer_list, shadow_layer_list


def get_composite_layer(input_image, df):
  base_layer_list = split_img_df(df, show=False)

  org_df = rgba2df(input_image)

  org_df["r"] = org_df["r"].apply(lambda x:int(x))
  org_df["g"] = org_df["g"].apply(lambda x:int(x))
  org_df["b"] = org_df["b"].apply(lambda x:int(x))

  org_df["diff_r"] = df["r"] - org_df["r"]
  org_df["diff_g"] = df["g"] - org_df["g"]
  org_df["diff_b"] = df["b"] - org_df["b"]
  
  org_df["shadow_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] >= 0 and x["diff_g"] >= 0 and x["diff_b"] >= 0 else False,
    axis=1
  )
  org_df["screen_flg"] = org_df.apply(
    lambda x: True if x["diff_r"] < 0 and x["diff_g"] < 0 and x["diff_b"] < 0 else False,
    axis=1
  )
    

  shadow_df = org_df.copy()
  shadow_df["a"] = org_df.apply(lambda x: 255 if x["shadow_flg"] == True else 0, axis=1)
  
  shadow_df["r"] = shadow_df["r"].apply(lambda x: x*255)
  shadow_df["g"] = shadow_df["g"].apply(lambda x: x*255)
  shadow_df["b"] = shadow_df["b"].apply(lambda x: x*255)

  shadow_df["r"] = (shadow_df["r"])/df["r"]
  shadow_df["g"] = (shadow_df["g"])/df["g"]
  shadow_df["b"] = (shadow_df["b"])/df["b"]
  
  shadow_df["label"] = df["label"]
  shadow_layer_list = split_img_df(shadow_df, show=True)

  screen_df = org_df.copy()

  screen_df["a"] = screen_df["screen_flg"].apply(lambda x: 255 if x == True else 0)

  screen_df["r"] = (screen_df["r"] - df["r"])/(1 - df["r"]/255) 
  screen_df["g"] = (screen_df["g"] - df["g"])/(1 - df["g"]/255) 
  screen_df["b"] = (screen_df["b"] - df["b"])/(1 - df["b"]/255) 

  screen_df["label"] = df["label"]
  screen_layer_list = split_img_df(screen_df, show=True)

  
  addition_df = org_df.copy()
  addition_df["a"] = addition_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  addition_df["r"] = org_df["r"] - df["r"] 
  addition_df["g"] = org_df["g"] - df["g"] 
  addition_df["b"] = org_df["b"] - df["b"]  

  addition_df["r"] = addition_df["r"].apply(lambda x: 0 if x < 0 else x)
  addition_df["g"] = addition_df["g"].apply(lambda x: 0 if x < 0 else x)
  addition_df["b"] = addition_df["b"].apply(lambda x: 0 if x < 0 else x)

  addition_df["label"] = df["label"]

  addition_layer_list = split_img_df(addition_df, show=True)

  subtract_df = org_df.copy()
  subtract_df["a"] = subtract_df.apply(lambda x: 255 if x["screen_flg"] == False and x["shadow_flg"] == False else 0, axis=1)

  subtract_df["r"] = df["r"] - org_df["r"]   
  subtract_df["g"] = df["g"] - org_df["g"] 
  subtract_df["b"] = df["b"] - org_df["b"]

  subtract_df["r"] = subtract_df["r"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["g"] = subtract_df["g"].apply(lambda x: 0 if x < 0 else x)
  subtract_df["b"] = subtract_df["b"].apply(lambda x: 0 if x < 0 else x)

  subtract_df["label"] = df["label"]

  subtract_layer_list = split_img_df(subtract_df, show=True)


  return base_layer_list, shadow_layer_list, screen_layer_list, addition_layer_list, subtract_layer_list
