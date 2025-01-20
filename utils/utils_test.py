import torch
from torch.nn import functional as F
import cv2
import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from skimage import measure
from statistics import mean
from scipy.ndimage import gaussian_filter
import warnings
import glob

import os
import OpenEXR
import Imath
import torchvision.transforms as transforms
from tqdm import tqdm

def save_exr(image_tensor, filename):
    image_np = image_tensor.numpy().transpose(1, 2, 0)
    height, width, channels = image_np.shape
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    if channels == 1:
        header['channels'] = {'Y': half_chan}
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'Y': image_np[:, :, 0].astype(np.float16).tostring()})
    elif channels == 3:
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
        exr = OpenEXR.OutputFile(filename, header)
        exr.writePixels({'R': image_np[:, :, 0].astype(np.float16).tostring(),
                         'G': image_np[:, :, 1].astype(np.float16).tostring(),
                         'B': image_np[:, :, 2].astype(np.float16).tostring()})
    exr.close()

warnings.filterwarnings('ignore')
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size])
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list

def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    # return np.uint8(255 * cam)
    return cam

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap



def evaluation_multi_proj(encoder,proj,bn, decoder, dataloader, device, out_path=None):
    transform = transforms.ToTensor()
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    with torch.no_grad():
        for i, (img, gt, label, _, _) in enumerate(tqdm(dataloader)):

            img = img.to(device)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            gt[gt > 0.5] = 1
            gt[gt <= 0.5] = 0
            if label.item()!=0:
                aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                              anomaly_map[np.newaxis,:,:]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            pr_list_sp.append(np.max(anomaly_map))

            if out_path is not None:
                cam = show_cam_on_image(img[0].cpu(), anomaly_map)
                cam = torch.from_numpy(cam)
                save_exr(cam, os.path.join(out_path, f'cam_{i}.exr'))

                hitmap = cvt2heatmap(anomaly_map*255)/255.0
                hitmap = torch.from_numpy(hitmap).permute(2, 0, 1)

                save_exr(hitmap, os.path.join(out_path, f'hitmap_{i}.exr'))
                color = img[0].cpu()
                # color = color - color.min()
                # color = color / color.max()
                save_exr(color, os.path.join(out_path, f'img_{i}.exr'))
                save_exr(gt[0].cpu(), os.path.join(out_path, f'gt_{i}.exr'))
                admap = transform(anomaly_map)
                save_exr(admap, os.path.join(out_path, f'pred_{i}.exr'))

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
    return auroc_px, auroc_sp, round(np.mean(aupro_list),4)

def generate_pred_file(encoder, proj,bn, decoder, path, img_transform, device):
    transform = transforms.ToTensor()
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    with torch.no_grad():
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img/255., (256, 256))

        img_input = img_transform(img)
        img_input = img_input.to(device).unsqueeze(0)
        inputs = encoder(img_input)
        features = proj(inputs)
        outputs = decoder(bn(features))
        anomaly_map, _ = cal_anomaly_map(inputs, outputs, img_input.shape[-1], amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        
        hitmap = cvt2heatmap(anomaly_map*255)/255.0
        hitmap = torch.from_numpy(hitmap).permute(2, 0, 1)

        return transform(img), img_input[0].cpu(), hitmap, transform(anomaly_map)

def generate_pred(encoder,proj,bn, decoder, path, img_transform, device, out_path=None):
    transform = transforms.ToTensor()
    encoder.eval()
    proj.eval()
    bn.eval()
    decoder.eval()
    with torch.no_grad():
        path = sorted(glob.glob(os.path.join(path) + "/*.png"))
        print(len(path))
        for i, p in enumerate(tqdm(path)):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img/255., (256, 256))
            img = img_transform(img)
            img = img.to(device).unsqueeze(0)
            inputs = encoder(img)
            features = proj(inputs)
            outputs = decoder(bn(features))
            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            if out_path is not None:
                cam = show_cam_on_image(img[0].cpu(), anomaly_map)
                cam = torch.from_numpy(cam)
                save_exr(cam, os.path.join(out_path, f'cam_{i}.exr'))

                hitmap = cvt2heatmap(anomaly_map*255)/255.0
                hitmap = torch.from_numpy(hitmap).permute(2, 0, 1)
                save_exr(hitmap, os.path.join(out_path, f'hitmap_{i}.exr'))

                color = img[0].cpu()
                # color = color - color.min()
                # color = color / color.max()
                save_exr(color, os.path.join(out_path, f'img_{i}.exr'))
                admap = transform(anomaly_map)
                save_exr(admap, os.path.join(out_path, f'pred_{i}.exr'))

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:

    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

#     df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    d = {'pro':[], 'fpr':[],'threshold': []}
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

#         df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        d['pro'].append(mean(pros))
        d['fpr'].append(fpr)
        d['threshold'].append(th)
    df = pd.DataFrame(d)
    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc
