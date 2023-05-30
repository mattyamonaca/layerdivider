from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import copy
from PIL import Image
import pickle

def get_mask_generator(pred_iou_thresh, stability_score_thresh, crop_n_layers, crop_n_points_downscale_factor, min_mask_region_area, model_path):
    sam_checkpoint = f"{model_path}/segment_model/sam_vit_h_4b8939.pth"
    device = "cuda"
    model_type = "default"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
    return mask_generator

def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks

def show_anns(image, masks, output_dir):
    if len(masks) == 0:
        return
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # pickle化してファイルに書き込み
    with open(f'{output_dir}/tmp/seg_layer/sorted_masks.pkl', 'wb') as f:
        pickle.dump(sorted_masks, f)
    polygons = []
    color = []
    mask_list = []
    for mask in sorted_masks:
        m = mask['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        img = np.dstack((img*255, m*255*0.35))
        img = img.astype(np.uint8)
        
        mask_list.append(img)
    
    base_mask = image 
    for mask in mask_list:
        base_mask = Image.alpha_composite(base_mask, Image.fromarray(mask))

    return base_mask

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)
