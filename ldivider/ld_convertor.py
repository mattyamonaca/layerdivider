import cv2
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from skimage import color 
from PIL import Image


def skimage_rgb2lab(rgb):
    return color.rgb2lab(rgb.reshape(1,1,3))


def rgb2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
  })
  return df

def mask2df(mask):
  h, w = mask.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  flg = mask.astype(int)
  df = pd.DataFrame({
      "x_l_m": x_l.ravel(),
      "y_l_m": y_l.ravel(),
      "m_flg": flg.ravel(),
  })
  return df


def rgba2df(img):
  h, w, _ = img.shape
  x_l, y_l = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
  r, g, b, a = img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]
  df = pd.DataFrame({
      "x_l": x_l.ravel(),
      "y_l": y_l.ravel(),
      "r": r.ravel(),
      "g": g.ravel(),
      "b": b.ravel(),
      "a": a.ravel()
  })
  return df

def hsv2df(img):
    x_l, y_l = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), indexing='ij')
    h, s, v = np.transpose(img, (2, 0, 1))
    df = pd.DataFrame({'x_l': x_l.flatten(), 'y_l': y_l.flatten(), 'h': h.flatten(), 's': s.flatten(), 'v': v.flatten()})
    return df

def df2rgba(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  a_img = img_df.pivot_table(index="x_l", columns="y_l",values= "a").reset_index(drop=True).values
  df_img = np.stack([r_img, g_img, b_img, a_img], 2).astype(np.uint8)
  return df_img

def df2bgra(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  a_img = img_df.pivot_table(index="x_l", columns="y_l",values= "a").reset_index(drop=True).values
  df_img = np.stack([b_img, g_img, r_img, a_img], 2).astype(np.uint8)
  return df_img

def df2rgb(img_df):
  r_img = img_df.pivot_table(index="x_l", columns="y_l",values= "r").reset_index(drop=True).values
  g_img = img_df.pivot_table(index="x_l", columns="y_l",values= "g").reset_index(drop=True).values
  b_img = img_df.pivot_table(index="x_l", columns="y_l",values= "b").reset_index(drop=True).values
  df_img = np.stack([r_img, g_img, b_img], 2).astype(np.uint8)
  return df_img

def pil2cv(image):
  new_image = np.array(image, dtype=np.uint8)
  if new_image.ndim == 2:
      pass
  elif new_image.shape[2] == 3:
      new_image = new_image[:, :, ::-1]
  elif new_image.shape[2] == 4:
      new_image = new_image[:, :, [2, 1, 0, 3]]
  return new_image

def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = new_image[:, :, ::-1]
    elif new_image.shape[2] == 4:
        new_image = new_image[:, :, [2, 1, 0, 3]]
    new_image = Image.fromarray(new_image)
    return new_image



