import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
import glob

class prepareData(Dataset):
    """
    Skeleton for loading datasets in the episodic Fashion.

    Args:
        root_dir: path to dataset root dir
        split: specialized to the dataset, but usually train/val/test
    """
    
    def __init__(self, path='./dataPreprocessed/',view='coronal'):
        fileList=sorted([f for f in glob.glob(path+view+'/*') if os.path.isfile(f)])
        views=['sagital','frontal','coronal']
        
        self.files=[np.load(file,allow_pickle=True) for file in fileList]
        self.view=views.index(view)
        self.len=self.files[0].shape[self.view]
        
    def fetchImage(self,idx):
        
        if self.view==2:
            imgList=[file[:,:,idx] for file in self.files[:-1]]
            annot=self.files[-1][:,:,idx]
            
        if self.view==1:
            imgList=[file[:,idx,:] for file in self.files[:-1]]
            annot=self.files[-1][:,idx,:]
            
        if self.view==0:
            imgList=[file[idx,:,:] for file in self.files[:-1]]
            annot=self.files[-1][idx,:,:]
            
        return imgList, annot

    def __getitem__(self, idx):
        imgList, annot=self.fetchImage(idx)
        
        imgs=np.stack(imgList)
        
        return imgs, annot
        
    def __len__(self):
        return self.len
        
class NpToTensor(object):
    """
    Convert `np.array` to `torch.Tensor`, but not like `ToTensor()`
    from `torchvision` because we don't rescale the values.
    """

    def __call__(self, arr):
        return torch.from_numpy(np.ascontiguousarray(arr)).float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class SegToTensor(object):

    def __call__(self, seg):
        seg = torch.from_numpy(seg.astype(np.float)).float()
        return seg.unsqueeze(0)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    

class TransformData(Dataset):
    """
    Transform a dataset by registering a transform for every input and the
    target. Skip transformation by setting the transform to None.

    Take
        dataset: the `Dataset` to transform (which must be a `SegData`).
        input_transforms: list of `Transform`s for each input
        target_transform: `Transform` for the target image
    """

    def __init__(self, dataset, input_transforms=None, target_transform=None):
        #super().__init__(dataset)
        self.ds = dataset
        self.input_transforms = input_transforms
        self.target_transform = target_transform
        

    def __getitem__(self, idx):
        
        # extract data from inner dataset
        inputs, target = self.ds[idx]
        inputs=self.input_transforms(inputs)
        count=np.count_nonzero(target)
        
        if self.target_transform is not None:
            target =self.target_transform(target)
        
        # repackage data
        return inputs, target, count

    def __len__(self):
        return len(self.ds)
    
    
def prepare_data(path, view, testSplit=0.2, valSplit=0.2, batch_size=1):      
    
    # load the data
    ds=prepareData(path,view)
    
    # transforms for the input and target
    image_transform = NpToTensor()
    target_transform = SegToTensor()
    
    # apply transforms and get class frequency
    TransformedDS = TransformData(ds, input_transforms=image_transform, target_transform=target_transform)
    
    # get the size of the splits   
    testSize= int(testSplit*len(TransformedDS))
    valSize= int(valSplit*len(TransformedDS))
    trainSize= len(TransformedDS)-testSize-valSize
 
    #split dataset here
    train_ds, test_ds, val_ds=random_split( TransformedDS ,[trainSize,testSize, valSize])
  
    return DataLoader(train_ds, batch_size=1, shuffle=True), DataLoader(test_ds, batch_size=1, shuffle=True), DataLoader(val_ds, batch_size=1, shuffle=True)
    
    
    
        