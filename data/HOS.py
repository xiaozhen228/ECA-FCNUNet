from scipy.io import loadmat,savemat
import numpy as np
from sklearn.decomposition import PCA
import os 
from data.base import FullImageDataset
SEED = 2333

def mean_std_normalize(img):
    mean = img.reshape(-1,img.shape[-1]).mean(axis=0)
    std = img.reshape(-1,img.shape[-1]).std(axis=0)   #标准化操作
    shape = [1] * img.ndim   #[1,1,1]
    shape[-1] = -1
    input_image = (img - mean.reshape(shape)) / std.reshape(shape)
    return input_image
class NewHOSDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 sample_percent=0.01,
                  batch_size=10,
                 batch_size_new = 20):
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        #========================================
        ynPCA = True
        #=======================================
        if ynPCA:
            if os.path.exists('./PCAdata/houston/Houston.mat'):
                im_mat = loadmat('./PCAdata/houston/Houston.mat')
                image = im_mat['Houston']
                print("加载PCAhouston成功")
            else:
                print("进行houstonPCA")
                im_mat = loadmat(image_mat_path)
                image = im_mat['Houston']
                image = self.applyPCA(image,50)    #103
        
            if not os.path.exists('./PCAdata/houston/Houston.mat'):
                savemat("./PCAdata/houston/Houston.mat",{'Houston':image})
        else:
            im_mat = loadmat(image_mat_path)
            image = im_mat['Houston']

        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['Houston_gt']

        
        image = mean_std_normalize(image)

        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size   #这里应该类似于定义了一个采样的策略
        super(NewHOSDataset, self).__init__(image, mask, training, np_seed=SEED,
                                                sample_percent=sample_percent,
                                                batch_size=batch_size,batch_size_new = batch_size_new)
    def applyPCA(self,X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        print('oldX.shape:', newX.shape)
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
        print('newX.shape:', newX.shape)
        return newX
    @property
    def num_classes(self):
        return 15