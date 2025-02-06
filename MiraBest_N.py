from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import urllib.request
import tarfile
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class MiraBest_N(data.Dataset):
    """
    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-N.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = 'N_batches'
    url = "http://www.jb.man.ac.uk/research/MiraBest/MiraBest_N/MiraBest_N_batches.tar.gz" 
    filename = "MiraBest_N_batches.tar.gz"
    tgz_md5 = '198337a1eb655a1f8642c590979d0e71'
    train_list = [  
                  ['data_batch_1', 'f6c591d62e166523403cd7cbdeb2b076'],
                  ['data_batch_2', '35e3302aa61a8dbbf20db3ad06e21b17'],
                  ['data_batch_3', '00d4725a4ba0f1cd6a298f9030673d3f'],
                  ['data_batch_4', '29f9d1784521012d7c3893daebd84bca'],
                  ['data_batch_5', 'cdfaafc3e6581f727f8fd13decc9b874'],
                  ['data_batch_6', 'a50f185ef0fbc8d3bfa2f7f4004e78ca'],
                  ['data_batch_7', '6d267af85775cc3f9d336dc11ecf6d7b'],
                  ]

    test_list = [
                 ['test_batch', 'b3ff79c3b6c2acf9a0865f8b3d71a35d'],
                 ]
    meta = {
                'filename': 'batches.meta',
                'key': 'label_names',
                'md5': 'e1b5450577209e583bc43fbf8e851965',
                }


    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = '/share/nas2_3/amahmoud/data/MiraBest'
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            if self.download:
                print("Dataset not found or corrupted. Attempting to download...")
                self._download()
        
                if not self._check_integrity():
                    raise RuntimeError("Download failed or dataset is still corrupted after download.")
            else:
                raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it.")


        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []
        self.filenames = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                    self.filenames.extend(entry['filenames'])
                else:
                    self.targets.extend(entry['fine_labels'])
                    self.filenames.extend(entry['filenames'])


        self.data = np.vstack(self.data).reshape(-1, 1, 150, 150)
        self.data = self.data.transpose((0, 2, 3, 1))

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = np.reshape(img,(150,150))
        img = Image.fromarray(img,mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            #print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

# ---------------------------------------------------------------------------------

class MBFRFull(MiraBest_N):
    
    """
        Child class to load all FRI (0) & FRII (1)
        [100, 102, 104, 110, 112] and [200, 201, 210]
        """
    
    def __init__(self, *args, **kwargs):
        super(MBFRFull, self).__init__(*args, **kwargs)
        
        fr1_list = [0,1,2,3,4]
        fr2_list = [5,6,7]
        exclude_list = [8,9]
        
        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()

print("MBFRFull class is defined and ready to be imported.")

# ---------------------------------------------------------------------------------

class MBFRConfident(MiraBest_N):

    """
    Child class to load only confident FRI (0) & FRII (1)
    [100, 102, 104] and [200, 201]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRConfident, self).__init__(*args, **kwargs)

        fr1_list = [0,1,2]
        fr2_list = [5,6]
        exclude_list = [3,4,7,8,9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()

# ---------------------------------------------------------------------------------

class MBFRUncertain(MiraBest_N):

    """
    Child class to load only uncertain FRI (0) & FRII (1)
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBFRUncertain, self).__init__(*args, **kwargs)

        fr1_list = [3,4]
        fr2_list = [7]
        exclude_list = [0,1,2,5,6,8,9]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            fr1 = np.array(fr1_list).reshape(1, -1)
            fr2 = np.array(fr2_list).reshape(1, -1)
            fr1_mask = (targets.reshape(-1, 1) == fr1).any(axis=1)
            fr2_mask = (targets.reshape(-1, 1) == fr2).any(axis=1)
            targets[fr1_mask] = 0 # set all FRI to Class~0
            targets[fr2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()

# ---------------------------------------------------------------------------------

class MBHybrid(MiraBest_N):

    """
    Child class to load confident(0) and uncertain (1) hybrid sources
    [110, 112] and [210]
    """

    def __init__(self, *args, **kwargs):
        super(MBHybrid, self).__init__(*args, **kwargs)

        h1_list = [8]
        h2_list = [9]
        exclude_list = [0,1,2,3,4,5,6,7]

        if exclude_list == []:
            return
        if self.train:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            targets[h1_mask] = 0 # set all FRI to Class~0
            targets[h2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            exclude_mask = ~(targets.reshape(-1, 1) == exclude).any(axis=1)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            targets[h1_mask] = 0 # set all FRI to Class~0
            targets[h2_mask] = 1 # set all FRII to Class~1
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()

# ---------------------------------------------------------------------------------
    
class MBRandom(MiraBest_N):

    """
    Child class to load 50 random FRI and 50 random FRII sources
    """

    def __init__(self, certainty='all', morphologies='all', *args, **kwargs):
        super(MBRandom, self).__init__(*args, **kwargs)
        
        # Checking flags
        # ------------------
        
        if certainty == 'certain':
            certainty_list1 = np.array([0, 1, 2])
            certainty_list2 = np.array([5, 6])
        elif certainty == 'uncertain':
            certainty_list1 = np.array([3, 4])
            certainty_list2 = np.array([7])
        else:
            certainty_list1 = np.array([0, 1, 2, 3, 4])
            certainty_list2 = np.array([5, 6, 7])   
        
        if morphologies == 'standard':
            morphology_list1 = np.array([0, 3]) 
            morphology_list2 = np.array([5, 7])   
        else:
            morphology_list1 = np.array([0, 1, 2, 3, 4]) 
            morphology_list2 = np.array([5, 6, 7])
           
        list_matches1 = np.in1d(certainty_list1, morphology_list1)
        list_matches2 = np.in1d(certainty_list2, morphology_list2)
        
        h1_list = certainty_list1[np.where(list_matches1)[0]]
        h2_list = certainty_list2[np.where(list_matches2)[0]]
        
        # ------------------
        
        if self.train:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0 # set all FRI to Class~0
            targets[h2_random] = 1 # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
        else:
            targets = np.array(self.targets)
            h1 = np.array(h1_list).reshape(1, -1)
            h2 = np.array(h2_list).reshape(1, -1)
            h1_mask = (targets.reshape(-1, 1) == h1).any(axis=1)
            h2_mask = (targets.reshape(-1, 1) == h2).any(axis=1)
            h1_indices = np.where(h1_mask)
            h2_indices = np.where(h2_mask)
            h1_random = np.random.choice(h1_indices[0], 50, replace=False)
            h2_random = np.random.choice(h2_indices[0], 50, replace=False)
            targets[h1_random] = 0 # set all FRI to Class~0
            targets[h2_random] = 1 # set all FRII to Class~1
            target_list = np.concatenate((h1_random, h2_random))
            exclude_mask = (targets.reshape(-1, 1) == target_list).any(axis=1)
            self.data = self.data[exclude_mask]
            self.targets = targets[exclude_mask].tolist()
                    
