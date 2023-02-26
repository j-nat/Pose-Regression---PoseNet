import os
import cv2
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing
        # cv2--> (H,W,C)
        ## torch tensor --> (C,H,W)
        # PIL --> (C,W,H)
        ## numpy --> (H,W,C)
        #(H,W,C)-(H,W,C)--> (C,H,W)



        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self, ):
        print("Computing mean image:")

        # TODO: Compute mean image

        # Initialize mean_image
        sum_image, mean_image = 0.0, 0.0

        # # Iterate over all training images
        # # Resize, Compute mean, etc...
        
        for new_img_path in self.images_path:

            image = Image.open(new_img_path)
            image = T.ToTensor()(image)
            image = T.Resize(self.resize)(image)
            image = np.asarray(image)
            sum_image += image

        mean_image = sum_image / len(self.images_path)

        np.save(self.mean_image_path, mean_image)
        print("Mean image computed!")

        return mean_image


    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = image = T.ToTensor()(data)
        data = T.Resize(self.resize)(data)
        data = torch.permute(data,(1,2,0))
        data = np.asarray(data)
        detect_mean = data - self.mean_image
        detect_mean = torch.from_numpy(detect_mean)
        detect_mean = torch.permute(detect_mean, (2, 0, 1))

        if self.train:
            result = T.RandomCrop(self.crop_size)(detect_mean)
        else:
            result = T.CenterCrop(self.crop_size)(detect_mean)

        result = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(result)
        result = result.to(torch.float32)

        return result, img_pose

    def __len__(self):
        return len(self.images_path)
