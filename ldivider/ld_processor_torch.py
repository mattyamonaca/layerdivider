
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle


from .ld_processor import calc_ciede
from .ld_processor_np import get_cls_update


def get_blur_torch(img: torch.Tensor, labels: torch.Tensor, size, blur=True):
    if blur:
        assert size % 2 == 1
        p = (size - 1) // 2
        img = F.pad(img, [p, p, p, p], mode='reflect')
        img = F.avg_pool2d(img, kernel_size=size, stride=1)
    
    cls = torch.unique(labels).reshape(-1, 1, 1, 1)
    masks = torch.bitwise_and(img[:, [3]] > 127, cls == labels)

    cls_counts = masks.sum(dim=(2, 3), keepdim=True) + 1e-7
    rgb_means = (img[:, :3] * masks).sum(dim=(2, 3), keepdim=True) / cls_counts
    
    rgb_means = rgb_means.squeeze().cpu().tolist()
    cls_list = cls.squeeze().cpu().tolist()
    cls_counts = cls_counts.squeeze().cpu().tolist()
    
    return rgb_means, cls_list, cls_counts, masks


def get_base_torch(img: np.ndarray, loop, cls_num, threshold, size, kmeans_samples=-1, device='cpu'):
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

    img_torch = rearrange([img], 'n h w c -> n c h w')
    img_torch = torch.from_numpy(img_torch).to(dtype=torch.float32, device=device)
    labels_torch = torch.from_numpy(labels.reshape((1, 1, im_h, im_w))).to(dtype=torch.float32, device=device)

    assert loop > 0
    img_torch_ori = img_torch.clone()
    for i in range(loop):
        rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size)
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
                mask = torch.bitwise_or(mask, cls2masks[srcc])
            labels_torch.masked_fill_(mask, tgtc)
            if i != loop - 1:
                for jj in range(3):
                    img_torch[:, jj].masked_fill_(mask[0], cls2rgb[tgtc][jj])

    cls_list = torch.unique(labels_torch)
    img_torch = img_torch_ori
    rgb_means, cls_list, cls_counts, masks = get_blur_torch(img_torch, labels_torch, size, blur=False)
    for mask, rgb in zip(masks, rgb_means):
        for jj in range(3):
            img_torch[:, jj][mask] = rgb[jj]
    
    img = rearrange(img_torch.cpu().numpy(), 'n c h w -> h w (n c)')
    img = img.clip(0, 255).astype(np.uint8)
    labels = labels_torch.cpu().numpy().squeeze().astype(np.uint32)
    return img, labels