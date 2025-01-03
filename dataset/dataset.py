from torchvision import transforms
from PIL import Image
import os
import torch
import glob
import numpy as np
from dataset.noise import Simplex_CLASS
import cv2

class ToTensor(object):
    def __call__(self, image):
        try:
            image = torch.from_numpy(image.transpose(2, 0,1))
        except:
            print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        return image
    
# 사용자 정의 변환: Numpy Array를 Resize
class ResizeNumpyArray:
    def __init__(self, output_size):
        self.output_size = output_size  # 원하는 크기 (width, height)

    def __call__(self, image):
        """
        Args:
            image (numpy.ndarray): Input image in numpy format.
        Returns:
            numpy.ndarray: Resized image.
        """
        if isinstance(image, np.ndarray):
            return cv2.resize(image, self.output_size)
        else:
            raise TypeError("Input must be a numpy array")

    
class Normalize(object):
    """
    Only normalize images
    """
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image

def get_data_transforms(size, isize):
    data_transforms = transforms.Compose([Normalize(),\
                    ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def get_inference_data_transforms(size, isize):
    data_transforms = transforms.Compose([
        ResizeNumpyArray((size, size)),
        Normalize(),
        ToTensor()])
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

class MVTecDataset_train(torch.utils.data.Dataset):
    def __init__(self, root, transform, synth_path = None, synth_num = 100):
        self.img_path = root
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        # load dataset
        self.img_paths = self.load_dataset()  # self.labels => good : 0, anomaly : 1
        
        if synth_path is None:
            self.synth_len = 0
            self.synth_num = 0
        else:
            self.synth_pass = sorted(glob.glob(os.path.join(synth_path, 'pass') + "/*.png"))
            self.synth_fail = sorted(glob.glob(os.path.join(synth_path, 'fail') + "/*.png"))
            self.synth_len = len(self.synth_pass)
            assert len(self.synth_pass) == len(self.synth_fail), "Synthetic pass and fail data should have the same number of images"
            for p, f in zip(self.synth_pass, self.synth_fail):
                assert os.path.basename(p) == os.path.basename(f), "Synthetic pass and fail data should have the same name"

            if synth_num < 0:
                self.synth_num = self.synth_len
            else:
                self.synth_num = min(synth_num, self.synth_len)

        print(f"Real data: {len(self.img_paths)}, Synthetic data: {self.synth_len}, Training Synthetic data: {self.synth_num}")

    def load_dataset(self):
        img_paths = glob.glob(os.path.join(self.img_path, 'good') + "/*.png")
        return img_paths

    def __len__(self):
        return self.synth_num + len(self.img_paths)

    def __getitem__(self, idx):

        synth_idx = 0
        if idx < self.synth_num:
            synth_idx = np.random.randint(0, self.synth_len)
            img_path = self.synth_pass[synth_idx]
        else:
            # print(idx, self.synth_len, len(self.img_paths))
            img_path = self.img_paths[idx-self.synth_num]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img= cv2.resize(img/255., (256, 256))
        ## Normal
        img_normal = self.transform(img)
        ## simplex_noise
        size = 256
        h_noise = np.random.randint(10, int(size//8))
        w_noise = np.random.randint(10, int(size//8))
        start_h_noise = np.random.randint(1, size - h_noise)
        start_w_noise = np.random.randint(1, size - w_noise)
        noise_size = (h_noise, w_noise)
        simplex_noise = self.simplexNoise.rand_3d_octaves((3, *noise_size), 6, 0.6)
        init_zero = np.zeros((256,256,3))
        init_zero[start_h_noise: start_h_noise + h_noise, start_w_noise: start_w_noise+w_noise, :] = 0.2 * simplex_noise.transpose(1,2,0)

        if idx < self.synth_num:
            img_noise = cv2.imread(self.synth_fail[synth_idx])
            img_noise = cv2.cvtColor(img_noise, cv2.COLOR_BGR2RGB)
            # 3D anomaly + noise
            # img_noise = cv2.resize(img_noise/255., (256, 256)) + init_zero
            # only 3D anomaly
            img_noise = cv2.resize(img_noise/255., (256, 256))
        else:
            img_noise = img + init_zero

        img_noise = self.transform(img_noise)

        return img_normal,img_noise,img_path.split('/')[-1]


class MVTecDataset_test(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform):
        self.img_path = os.path.join(root, 'test')
        self.gt_path = os.path.join(root, 'ground_truth')
        self.simplexNoise = Simplex_CLASS()
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img= cv2.resize(img, (256, 256))
        ## Normal
        img = self.transform(img)
        ## simplex_noise
        
        if gt == 0:
            gt = torch.zeros([1, img.shape[-1], img.shape[-1]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.shape[1:] == gt.shape[1:], f"image.size != gt.size !!!, {img.shape[1:]}, {gt.shape[1:]}"

        return (img, gt, label, img_type, img_path.split('/')[-1])



