
from torch.utils.data import dataset
import numpy as np

import torch
from torch.utils import data
import random 
SEED = 2000
def divisible_pad(image_list, size_divisor=128, to_tensor=True):
    """

    Args:
        image_list: a list of images with shape [channel, height, width]
        size_divisor: int
        to_tensor: whether to convert to tensor
    Returns:
        blob: 4-D ndarray of shape [batch, channel, divisible_max_height, divisible_max_height]
    """
    max_shape = np.array([im.shape for im in image_list]).max(axis=0)

    max_shape[1] = int(np.ceil(max_shape[1] / size_divisor) * size_divisor)
    max_shape[2] = int(np.ceil(max_shape[2] / size_divisor) * size_divisor)

    if to_tensor:
        storage = torch.FloatStorage._new_shared(len(image_list) * np.prod(max_shape))
        out = torch.Tensor(storage).view([len(image_list), max_shape[0], max_shape[1], max_shape[2]])
        out = out.zero_()
    else:
        out = np.zeros([len(image_list), max_shape[0], max_shape[1], max_shape[2]], np.float32)

    for i, resized_im in enumerate(image_list):
        out[i, :, 0:resized_im.shape[1], 0:resized_im.shape[2]] = torch.from_numpy(resized_im)

    return out

class FullImageDataset(dataset.Dataset):   #就是这个，返回应该在这里看
    def __init__(self,
                 image,
                 mask,
                 training,
                 np_seed=2000,
                 sample_percent=0.01,
                 batch_size=10,
                 batch_size_new =20,
                 ):
        self.image = image
        self.mask = mask
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size
        self.batch_size_new = batch_size_new
        self._seed = np_seed
        self.per_class = 20
        self._rs = np.random.RandomState(np_seed)

        # set list lenght = 9999 to make sure seeds enough
        self.seeds_for_minibatchsample = [e for e in self._rs.randint(low=2 << 31 - 1, size=9999)]
        self.preset()

    def preset(self):
        if self.training:

            #print(self.num_train_samples_per_class)
            #至两个是啥你先得知道
            train_indicator, test_indicator = fixed_num_sample(self.mask, self.sample_percent,self.num_classes, self._seed,self.per_class)

            blob = divisible_pad([np.concatenate([self.image.transpose(2, 0, 1),
                                              self.mask[None, :, :],
                                              train_indicator[None, :, :],
                                              test_indicator[None, :, :]], axis=0)], 16, False) #这里这个16是类别数
            im = blob[0, :self.image.shape[-1], :, :]    #(204,512,224)
            #这里是按照通道拼接的，只是将纬度转化为了160的目的是什么，拿到是为了便于输入吗？应该类似于一个padding操作
            mask = blob[0, -3, :, :]
            self.train_indicator = blob[0, -2, :, :]
            self.test_indicator = blob[0, -1, :, :]     


            self.train_inds_list = minibatch_sample(mask, self.train_indicator, self.batch_size,
                                                    seed=self.seeds_for_minibatchsample.pop())
            self.train_inds_list_new = minibatch_sample(mask, self.test_indicator, self.batch_size_new,
                                                    seed=self.seeds_for_minibatchsample.pop())
            
            #mask应该是所有的标签
            self.pad_im = im
            self.pad_mask = mask
            np.save('train_indicator.npy', self.train_indicator)
            np.save('test_indicator.npy', self.test_indicator)
            np.save('pad_im.npy', self.pad_im)
            np.save('pad_mask.npy', self.pad_mask)
            print("训练集",np.sum(self.train_indicator))
            print("测试集",np.sum(self.test_indicator))

        else: #这是如果是测试状态怎们办
            self.train_indicator,self.test_indicator =np.load('train_indicator.npy'), np.load('test_indicator.npy')
            self.pad_im = np.load('pad_im.npy')
            self.pad_mask = np.load('pad_mask.npy')

            


    def resample_minibatch(self):
        self.train_inds_list = minibatch_sample(self.pad_mask, self.train_indicator, self.batch_size,
                                                seed=self.seeds_for_minibatchsample.pop())

    @property
    def num_classes(self):
        return 16

    def __getitem__(self, idx):  #你看这里用了这个idx，因为训练时候是小批量，测试的时候没猜错应该是用的全部去测试
        #这里是和dataloader中的sampler相联系的，也就是sampler产生这这里的index
        if self.training:  
            return self.pad_im, self.pad_mask, self.train_inds_list[idx],self.train_inds_list_new[idx]   #返回了这样三样东西，方法覆盖了

        else:
            return self.pad_im, self.pad_mask, self.test_indicator,self.train_indicator

    def __len__(self):
        if self.training:
            return len(self.train_inds_list)   #这是返回有训练集的时候
        else:
            return 1    #返回以就是只有测试集的情况


class MinibatchSampler(data.Sampler):   #这里是印度松数据集定义的采样方法,dataloader中的一个迭代器
    def __init__(self, dataset: FullImageDataset):
        super(MinibatchSampler, self).__init__(None)
        self.dataset = dataset
        self.g = torch.Generator()
        self.g.manual_seed(SEED)

    def __iter__(self):
    #    self.dataset.resample_minibatch()
        n = len(self.dataset)   #10
        return iter(torch.randperm(n, generator=self.g).tolist())   #这里生成的不就是index吗？

    def __len__(self):    #感觉好像无关紧要
        return len(self.dataset)

'''
def fixed_num_sample(gt_mask: np.ndarray, sample_percent, num_classes, seed=2000):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    r=random.random
    random.seed(seed)
    #(106，610，340)
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    shuchu=[]

    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        count=np.sum(gt_mask_flatten == i)
        num_train_samples=np.ceil(count*sample_percent)   #采样训练样本的个数
        num_train_samples = num_train_samples.astype(np.int32) 
        if num_train_samples <5:       #这里是加入了类别不平衡采样策略，就是每类样本都采样这样一定的比例
            num_train_samples=5 # At least 5 samples per class
        random.shuffle(inds,random = r)

        train_inds = inds[:num_train_samples]
        test_inds = inds[num_train_samples:]
        shuchu.extend(train_inds)

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1
        #print("shuchu",shuchu)
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    return train_indicator, test_indicator
'''
def fixed_num_sample(gt_mask: np.ndarray, sample_percent, num_classes, seed=2000,per_class=5):
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        num_train_samples: int
        num_classes: scalar
        seed: int

    Returns:
        train_indicator, test_indicator
    """
    r=random.random
    random.seed(seed)
    #(106，610，340)
    gt_mask_flatten = gt_mask.ravel()
    train_indicator = np.zeros_like(gt_mask_flatten)
    test_indicator = np.zeros_like(gt_mask_flatten)
    shuchu=[]
    print("每类有标签采样：",per_class)

    for i in range(1, num_classes + 1):
        inds = np.where(gt_mask_flatten == i)[0]
        count=np.sum(gt_mask_flatten == i)
        num_train_samples= per_class  #采样训练样本的个数
        #num_train_samples = num_train_samples.astype(np.int32) 
        if num_train_samples <0:       #这里是加入了类别不平衡采样策略，就是每类样本都采样这样一定的比例
            num_train_samples=5 # At least 5 samples per class
        random.shuffle(inds,random = r)

        train_inds = inds[:num_train_samples]   # 这里只是划分训练集和测试集
        test_inds = inds[num_train_samples:]
        shuchu.extend(train_inds)

        train_indicator[train_inds] = 1
        test_indicator[test_inds] = 1
        #=================================
        #test_indicator[train_inds] = 1    # 这是将训练集也算入最后的计算指标
        #=================================
        #print("shuchu",shuchu)
    train_indicator = train_indicator.reshape(gt_mask.shape)
    test_indicator = test_indicator.reshape(gt_mask.shape)
    return train_indicator, test_indicator



def minibatch_sample(gt_mask: np.ndarray, train_indicator: np.ndarray, batch_size, seed):  #这里采样的是每一个batchsize的序号
    """

    Args:
        gt_mask: 2-D array of shape [height, width]
        train_indicator: 2-D array of shape [height, width]
        minibatch_size:

    Returns:

    """
    r=random.random
    random.seed(seed)
    # split into N classes
    cls_list = np.unique(gt_mask)   #(160 * 160) 该函数是去除数组中的重复数字，并进行排序之后输出，这是个一维向量
    inds_dict_per_class = dict()
    print("每批量每类:",batch_size)
    for cls in cls_list:
        train_inds_per_class = np.where(gt_mask == cls, train_indicator, np.zeros_like(train_indicator))#这三量的纬度一定要一致，然后一定要根据第一个条件从后面两个矩阵中选择
        inds = np.where(train_inds_per_class.ravel() == 1)[0]
        random.shuffle(inds,random=r)   

        inds_dict_per_class[cls] = inds  #这是一个字典，存储每一类的序号

    train_inds_list = []
    cnt = 0
    while True:
        train_inds = np.zeros_like(train_indicator).ravel()
        for cls, inds in inds_dict_per_class.items():
                random.shuffle(inds,random=r)
                cd=min(batch_size, len(inds))  #然后这里是每一批要不采batchsize个，要不是该类训练集的总数
                fetch_inds = inds[:cd]   #这里是取了5
                train_inds[fetch_inds] = 1    #每一类取5个的意思

        cnt += 1
        if cnt == 11:
            print(cnt)
            return train_inds_list
        train_inds_list.append(train_inds.reshape(train_indicator.shape))   #他xiang相当于采了10批，属于随机采样

