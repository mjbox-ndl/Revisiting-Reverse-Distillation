import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from utils.utils_test import evaluation_multi_proj, generate_pred
from utils.utils_train import MultiProjectionLayer
from dataset.dataset import MVTecDataset_test, get_data_transforms, get_inference_data_transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_folder', default = './your_checkpoint_folder', type=str)
    parser.add_argument('--data_folder', default = './your_dataset_path', type=str)
    parser.add_argument('--image_size', default = 256, type=int)
    parser.add_argument('--classes', nargs="+", default=["carpet", "leather"])
    parser.add_argument('--output_folder', default=None, type=str)
    pars = parser.parse_args()
    return pars

def load_model(_class_, device, pars):
    # Use pretrained wide_resnet50 for encoder
    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)

    bn = bn.to(device)
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    proj_layer =  MultiProjectionLayer(base=64).to(device)
    # Load trained weights for projection layer, bn (OCBE), decoder (student)    
    checkpoint_class = os.path.join(pars.checkpoint_folder, _class_, f'wres50_{_class_}.pth')
    ckp = torch.load(checkpoint_class, map_location='cpu')
    proj_layer.load_state_dict(ckp['proj'])
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    return encoder, proj_layer, bn, decoder

def inference(_class_, pars):
    if not os.path.exists(pars.checkpoint_folder):
        os.makedirs(pars.checkpoint_folder)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_transform, gt_transform = get_data_transforms(pars.image_size, pars.image_size)
    
    test_path           = os.path.join(pars.data_folder, _class_)

    test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, proj_layer, bn, decoder = load_model(_class_, device, pars)
  
    auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device)        
    print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_, auroc_sp, auroc_px, aupro_px))
    return auroc_sp, auroc_px, aupro_px

def predict(pars, is_real=True):
    _class_ = pars.classes
    print(f'Start prediction for class: {_class_} ...\n')
    setup_seed(111)

    if not os.path.exists(pars.output_folder):
        os.makedirs(pars.output_folder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = pars.image_size
    data_transform, gt_transform = get_inference_data_transforms(size, size)
    
    print('\tLoading model...')
    encoder, proj_layer, bn, decoder = load_model(_class_, device, pars)

    if is_real:
        test_path = os.path.join(pars.data_folder, _class_)
        test_data = MVTecDataset_test(root=test_path, transform=data_transform, gt_transform=gt_transform)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

        auroc_px, auroc_sp, aupro_px = evaluation_multi_proj(encoder, proj_layer, bn, decoder, test_dataloader, device, out_path=pars.output_folder)
        print(f'{_class_} AUROC: {auroc_px:.4f} AUROC_SP: {auroc_sp:.4f} AUPRO: {aupro_px:.4f}')
        print('\n... Done prediction for class:', _class_)
        return auroc_px, auroc_sp, aupro_px
    else:
        print('\tGenerating results for synthetic data...')
        generate_pred(encoder, proj_layer, bn, decoder, pars.data_folder, data_transform, device, out_path=pars.output_folder)
        print('\n... Done prediction')
        return None

if __name__ == '__main__':
    pars = get_args()

    item_list = [ 'carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    for c in pars.classes:
        auroc_sp, auroc_px, aupro_px = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'{pars.checkpoint_folder}/metrics_checkpoints.csv', index=False)