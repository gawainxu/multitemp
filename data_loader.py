from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.datasets.utils import extract_archive
import dataUtil
import numpy as np
import os
import gzip
import torch
import struct
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from util import TwoCropTransform, common_elements
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import torch.utils.data as data


class iCIFAR10(CIFAR10):
    def __init__(self, root, classes=range(10), train=True, transform=None,
                 target_transform=None, download=False, label_dict = None):
        super(iCIFAR10, self).__init__(root,
                                       train=train,
                                       transform=transform,
                                       target_transform=target_transform,
                                       download=download)
        self.label_dict = label_dict

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

            print("Final Data Size ", len(self.train_data))

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class iCIFAR100(CIFAR100):
    def __init__(self, root,
                 classes=range(100),
                 superClass = None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 label_dict = None):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.label_dict = label_dict

        if superClass is not None:
            classes = [dataUtil.classMap[n] for n in dataUtil.superClasses[superClass]] 

        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels
            
        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels        


class mnist(MNIST):
    
    def __init__(self, root,
                 classes=range(10),
                 train = True,
                 transform = None,
                 target_transform = None,
                 download = True,
                 label_dict = None, last_features_list=None, 
                 last_feature_labels_list=None, last_model=None, 
                 subsample_transform=None, portion_out=0.1, upsample_times=1):
        super(mnist, self).__init__(root, train=train,
                                    transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        
        self.label_dict = label_dict
        # Select subset of classes
        if self.train:
            train_data = []
            train_labels = []

            for i in range(0, len(self.data)):    # subsample
                if self.targets[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.targets[i])

            self.traindata = torch.stack(train_data).numpy()
            self.trainlabels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(0, len(self.data)):
                if self.targets[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.targets[i])   # it is torch tensor !!!!!!!!!!!!!

            print(len(test_data))
            self.testdata = torch.stack(test_data).numpy()
            self.testlabels = test_labels
        
        
    def __getitem__(self, index):
        if self.train:
            img, target = self.traindata[index], self.trainlabels[index]
        else:
            img, target = self.testdata[index], self.testlabels[index]

        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target.item())]

        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.traindata)
        else:
            return len(self.testdata)
        
        
    def get_image_class(self, label):
        return self.traindata[np.array(self.trainlabels) == label]
    
    
class mnist1(data.Dataset):


      datas = ['train-images-idx3-ubyte.gz',
               'train-labels-idx1-ubyte.gz',
               't10k-images-idx3-ubyte.gz',
               't10k-labels-idx1-ubyte.gz']

      taining_file = 'training.pt'
      test_file = 'test.pt'

      def __init__(self, root, classes=range(10), train=True, download=False,
                   transform=None, target_transform=None):
   
          self.root = os.path.expanduser(root)
          self.train = train
          self.transform = transform
          self.target_transform = target_transform

          if self.train:
             self.train_data = self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.train_labels = self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')
             
             train_data = []
             train_labels = []

             for i in range(0, len(self.train_data)):
                 if self.train_labels[i] in classes:
                     train_data.append(self.train_data[i])
                     train_labels.append(self.train_labels[i])

             self.train_data = torch.stack(train_data).numpy()
             self.train_labels = train_labels
             
          else:
             self.test_data =  self.read_img_file(root + '/mnist/raw/train-images-idx3-ubyte.gz')
             self.test_labels =  self.read_label_file(root + '/mnist/raw/train-labels-idx1-ubyte.gz')             
             
             test_data = []
             test_labels = []

             for i in range(len(self.test_data), 10):
                 if self.test_labels[i] in classes:
                     test_data.append(self.test_data[i])
                     test_labels.append(self.test_labels[i])   # it is torch tensor !!!!!!!!!!!!!

             print(len(test_data))
             self.test_data = torch.stack(test_data).numpy()
             self.test_labels = test_labels

       
      def __getitem__(self, index):
          """
          Args:
              index(int): Index
          Returns:
              tuple: (image, target) where target is index of the target class
          """
          if self.train:
             img, target = self.train_data[index], self.train_labels[index]
          else:
             img, target = self.test_data[index], self.test_labels[index]

          if self.transform is not None:
             img = Image.fromarray(np.squeeze(img), mode='L')
             img = self.transform(img)

          if self.target_transform is not None:
             target = self.target_transform(target)

          return img, target


      def __len__(self):
          if self.train:
             return len(self.train_data)
          else:
             return len(self.test_data) 
         
      def get_image_class(self, label):
          if self.train:
              return self.train_data[np.array(self.train_labels) == label]
          else:
              return self.test_data[np.array(self.test_labels) == label]
         
            
      def read_label_file(self, path):
          # read all images at once  
          # return: torch tensor  
          
          f = gzip.open(path, 'rb')
          f.read(8)                  # 4 byte integer magic number, 4 byte number of items
          if self.train:
              num_imgs = 10000        #struct.unpack('>I', f.read(4))[0]  !!!!!!!!!!!
          else:
              num_imgs = 10000
    
          buf = f.read(num_imgs)
          label = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          label_torch = torch.from_numpy(label).view(num_imgs).long()

          return label_torch
      
        
      def read_img_file(self, path):
          # read all images at once
          # return: torch tensor
          f = gzip.open(path, 'rb')
          f.read(8)                                         # skip the first 16 bytes, 4 byte integer magic number, 
          if self.train:
              num_imgs = 10000                                  #struct.unpack('>I', f.read(4))[0]       # 4 byte integer number of images, !!!!!!!!!!!!!!! 
          else:
              num_imgs = 10000                  
          num_rows = struct.unpack('>I', f.read(4))[0]      # 4 byte number of rows
          num_cols = struct.unpack('>I', f.read(4))[0]      # 4 byte number of columns
          buf = f.read(num_cols*num_rows*num_imgs)
          data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
          #data.reshape(num_images, image_size, image_size, 1)
          data_torch = torch.from_numpy(data).view(num_imgs, 1, num_rows, num_cols)
          return data_torch
                  

class TinyImagenet(Dataset):
    """
    Defines Tiny Imagenet as for the others pytorch datasets.
    """
    def __init__(self, root, classes=range(200), train=True, transform=None,
                 target_transform=None, download=False, label_dict = None):
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.label_dict = label_dict

        if download:
            if os.path.isdir(root) and len(os.listdir(root)) > 0:
                print('Download not needed, files already on disk.')
            else:
                from google_drive_downloader import GoogleDriveDownloader as gdd

                # https://drive.google.com/file/d/1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj/view
                print('Downloading dataset')
                gdd.download_file_from_google_drive(
                    file_id='1Sy3ScMBr0F4se8VZ6TAwDYF-nNGAAdxj',

                    dest_path=os.path.join(root, 'tiny-imagenet-processed.zip'),
                    unzip=True)

        self.data = []
        for num in range(20):                                       #20
            self.data.append(np.load(os.path.join(
                root, 'processed/x_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.data = np.concatenate(np.array(self.data))

        self.targets = []
        for num in range(20):                               # 20
            self.targets.append(np.load(os.path.join(
                root, 'processed/y_%s_%02d.npy' %
                      ('train' if self.train else 'val', num+1))))
        self.targets = np.concatenate(np.array(self.targets))
        
        train_data = []
        train_labels = []

        for i in range(len(self.data)):
            if self.targets[i] in classes:
                train_data.append(self.data[i])
                train_labels.append(self.targets[i])

        self.data = np.array(train_data)
        self.targets = train_labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(255 * img))                     ## put it in non transform ????????????
        original_img = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        if hasattr(self, 'logits'):
            return img, target, original_img, self.logits[index]

        return img, target
    
    def get_image_class(self, label):

        return self.data[np.array(self.targets) == label]


class customSVHN(SVHN):

    def __init__(self, root, train, classes=range(10), download=False, transform = None, target_transform = None, label_dict = None):
        super(customSVHN, self).__init__(root=root, split=train, transform = transform, target_transform = target_transform, download=download)
    
        self.root = root
        self.train = train
        self.label_dict = label_dict

        if self.split == "train":
            train_data = []
            train_labels = []

            for i in range(len(self.data)):
                if self.labels[i] in classes:
                    train_data.append(self.data[i])
                    train_labels.append(self.labels[i])

            self.train_data = np.array(train_data)
            self.train_labels = train_labels

        else:
            test_data = []
            test_labels = []

            for i in range(len(self.data)):
                if self.labels[i] in classes:
                    test_data.append(self.data[i])
                    test_labels.append(self.labels[i])

            self.test_data = np.array(test_data)
            self.test_labels = test_labels


    def __getitem__(self, index):

        if self.split == "train":
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target


    def __len__(self):

        if self.split == "train":
            return len(self.train_data)
        else:
            return len(self.test_data)


    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


    def append(self, images, labels):
        """Append dataset with images and labels

        Args:
            images: Tensor of shape (N, C, H, W)
            labels: list of labels
        """

        self.train_data = np.concatenate((self.train_data, images), axis=0)
        self.train_labels = self.train_labels + labels


class CUB(Dataset):

    def __init__(self, root, classes=range(200), download=False,
                 train=True, transform=None, target_transform=None,
                 label_dict=None):
        
        self.root = root
        self.transform = transform
        self.target_trasnform = target_transform
        self.train = train
        self.label_dict = label_dict
        self.loader = default_loader

        images_folder = "/CUB_200_2011/images/"
        image_list_file = "CUB_200_2011/images.txt"
        image_class_file = "CUB_200_2011/image_class_labels.txt"
        train_test_split_file = "CUB_200_2011/train_test_split.txt"

        with open(os.path.join(self.root, image_list_file), "r") as f:
            lines = f.readlines()

        self.file_dict = {"indices": [], "path": [], "label": [], "train_test_split": []}
        for line in lines:
            index, image_path = line.split(" ")
            index = int(index)
            image_path = image_path[:-1]
            self.file_dict["indices"].append(index)
            self.file_dict["path"].append(image_path)

        with open(os.path.join(self.root, image_class_file), "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            index, class_index = line.split(" ")
            index = int(index)
            class_index = int(class_index)
            assert i+1 == self.file_dict["indices"][i]
            self.file_dict["label"].append(class_index)

        with open(os.path.join(self.root, train_test_split_file), "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            index, split = line.split(" ")
            index = int(index)
            split = int(bool(10%5))
            assert i+1 == self.file_dict["indices"][i]
            self.file_dict["train_test_split"].append(split)

        if self.train:
            self.train_data = []
            self.train_labels = []
            for i, (path, label, split) in enumerate(zip(self.file_dict["path"], self.file_dict["label"], self.file_dict["train_test_split"])):
                if split == 0 and label not in classes:
                    continue
                img = self.loader(self.root + images_folder + path)
                self.train_data.append(img)
                self.train_labels.append(label)
                                                                                        
        else:
            self.test_data = []
            self.test_labels = []
            for i, (path, label, split) in enumerate(zip(self.file_dict["path"], self.file_dict["label"], self.file_dict["train_test_split"])):
                if split == 1 and label not in classes:
                    continue
                img = self.loader(self.root + images_folder + path)
                self.test_data.append(img)
                self.test_labels.append(label)


    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_trasnform is not None:
            target = self.target_trasnform(target)

        if self.label_dict is not None:
            target = self.label_dict[str(target)]

        return img, target

    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    
    def get_image_class(self, label):
        return self.train_data[np.array(self.train_labels) == label]
    
    
    def get_part_data(self, xidxs):
        
        self.train_data = np.delete(self.train_data, xidxs, 0)
        self.train_labels = np.delete(self.train_labels, xidxs, 0)


class Aircraft(VisionDataset):
    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')
    img_folder = os.path.join('fgvc-aircraft-2013b', 'data', 'images')

    def __init__(self, root, train=True, classes=range(100), download=False,  transform=None,
                 target_transform=None, label_dict = None, class_type='variant'):

        super(Aircraft, self).__init__(root, transform=transform, target_transform=target_transform)
        split = 'trainval' if train else 'test'
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))

        self.root = root
        self.class_type = class_type
        self.classes = classes
        self.split = split
        self.loader = default_loader
        self.classes_file = os.path.join(self.root, 'fgvc-aircraft-2013b', 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        (image_ids, targets, total_classes, class_to_idx) = self.find_classes()
        samples, targets = self.make_dataset(image_ids, targets)

        self.samples = samples
        self.targets = targets
        self.total_classes = total_classes
        self.class_to_idx = class_to_idx


    def __getitem__(self, index):
        sample, target = self.samples[index], self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.img_folder)) and \
               os.path.exists(self.classes_file)

    def find_classes(self):
        # read classes file, separating out image IDs and class names
        image_ids = []
        targets = []
        with open(self.classes_file, 'r') as f:
            for line in f:
                split_line = line.split(' ')
                image_ids.append(split_line[0])
                targets.append(' '.join(split_line[1:]))

        # index class names
        classes = np.unique(targets)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        targets = [class_to_idx[c] for c in targets]

        return image_ids, targets, classes, class_to_idx

    def make_dataset(self, image_ids, targets):
        assert (len(image_ids) == len(targets))
        images = []
        labels = []
        for i in range(len(image_ids)):
            if targets[i] not in self.classes:
                continue
            item = os.path.join(self.root, self.img_folder, '%s.jpg' % image_ids[i])
            image = self.loader(item)
            images.append(image)
            labels.append(targets[i])
        return images, labels





if __name__ == "__main__":
    transform = transforms.Compose([
       # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])                                      # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    # train_set = iCIFAR100(root='../datasets/', train=True,
    #                        classes=range(0, 10),
    #                        download=False, transform=None)
    # train_set = apply_transform(train_set, TwoCropTransform(transform))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=200, shuffle=True, num_workers=1)
    # for i, (img, l) in enumerate(train_loader):
    #     if i == 0:
    #         break
    root_path = "../datasets"
    dataset = customSVHN(root='../datasets', classes=[0])
    print(dataset[0][1])
    print(len(dataset))