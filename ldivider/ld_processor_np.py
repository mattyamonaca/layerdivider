import cv2
from einops import rearrange
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.utils import shuffle

from .ld_processor import calc_ciede


def get_cls_update(ciede_df, threshold, cls2counts):
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
        max_cls = max(merge, key=cls2counts.get)
        for cls in merge:
            if cls != max_cls:
                merge_dict[cls] = max_cls
    return merge_dict


def get_blur_np(img: np.ndarray, labels: np.ndarray, size, blur=True):
    
    if blur:
        img = rearrange(img, 'n c h w -> h w (n c)').astype(np.float32)
        img = cv2.blur(img, (size, size))
        img = rearrange(img, 'h w (n c) -> n c h w', n=1)
    
    cls = np.unique(labels).reshape(-1, 1, 1, 1)
    masks = np.bitwise_and(img[:, [3]] > 127, cls == labels)
    
    cls_counts = masks.sum(axis=(2, 3), keepdims=True) + 1e-10
    rgb_means = (img[:, :3] * masks).sum(axis=(2, 3), keepdims=True) / cls_counts

    rgb_means = rgb_means.squeeze().tolist()
    cls_list = cls.squeeze().tolist()
    cls_counts = cls_counts.squeeze().tolist()
    
    return rgb_means, cls_list, cls_counts, masks


def get_base_np(img: np.ndarray, loop, cls_num, threshold, size, debug=False, kmeans_samples=-1, device='cpu'):
    rgb_flatten = cluster_samples = img[..., :3].reshape((-1, 3))
    im_h, im_w = img.shape[:2]

    alpha_mask = np.where(img[..., 3] > 127)
    resampled = False
    if rgb_flatten.shape[0] > len(alpha_mask[0]):
        cluster_samples = img[..., :3][alpha_mask].reshape((-1, 3))
        resampled = True

    if len(rgb_flatten) > kmeans_samples and kmeans_samples > 0:
        cluster_samples = shuffle(cluster_samples, random_state=0, n_samples=kmeans_samples)
        resampled = True

    kmeans = MiniBatchKMeans(n_clusters=cls_num).fit(cluster_samples)
    if resampled:
        labels = kmeans.predict(rgb_flatten)
    else:
        labels = kmeans.labels_

    img_np = rearrange([img], 'n h w c -> n c h w').astype(np.float32)
    labels_np = labels.reshape((1, 1, im_h, im_w)).astype(np.float32)

    assert loop > 0
    img_np_ori = np.copy(img_np)
    for i in range(loop):
        rgb_means, cls_list, cls_counts, masks = get_blur_np(img_np, labels_np, size)
        ciede_df = calc_ciede(rgb_means, cls_list)
        cls2rgb, cls2counts, cls2masks = {}, {}, {}
        for c, rgb, count, mask in zip(cls_list, rgb_means, cls_counts, masks):
            cls2rgb[c] = rgb
            cls2counts[c] = count
            cls2masks[c] = mask[None, ...]

        merge_dict = get_cls_update(ciede_df, threshold, cls2counts)
        tgt2merge, notmerged = {}, set(cls_list)
        for k, v in merge_dict.items():
            if v not in tgt2merge:
                tgt2merge[v] = []
                notmerged.remove(v)
            tgt2merge[v].append(k)
            notmerged.remove(k)
        for k in notmerged:
            tgt2merge[k] = []
        
        for tgtc, srcc_list in tgt2merge.items():
            mask = cls2masks[tgtc]
            for srcc in srcc_list:
                mask = np.bitwise_or(mask, cls2masks[srcc])
            labels_np[mask] = tgtc
            if i != loop - 1:
                for jj in range(3):
                    img_np[:, jj][mask[0]] = cls2rgb[tgtc][jj]
        
    cls_list = np.unique(labels_np)
    img_np = img_np_ori
    rgb_means, cls_list, cls_counts, masks = get_blur_np(img_np, labels_np, size, blur=False)
    for mask, rgb in zip(masks, rgb_means):
        for jj in range(3):
            img_np[:, jj][mask] = rgb[jj]
    
    img = rearrange(np.clip(img_np, 0, 255), 'n c h w -> h w (n c)').astype(np.uint8)
    labels = labels_np.squeeze().astype(np.uint32)
    return img, labels