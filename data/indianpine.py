from scipy.io import loadmat
from data.base import FullImageDataset

SEED = 2333
def mean_std_normalize(img):
    mean = img.reshape(-1,img.shape[-1]).mean(axis=0)
    std = img.reshape(-1,img.shape[-1]).std(axis=0)   #标准化操作
    shape = [1] * img.ndim   #[1,1,1]
    shape[-1] = -1
    input_image = (img - mean.reshape(shape)) / std.reshape(shape)
    return input_image

class NewIndianPinesDataset(FullImageDataset):
    def __init__(self,
                 image_mat_path,
                 gt_mat_path,
                 training=True,
                 sample_percent=0.01,    
                 batch_size=10,
                 batch_size_new = 20):#只采样了1%的数据
        self.im_mat_path = image_mat_path
        self.gt_mat_path = gt_mat_path
        im_mat = loadmat(image_mat_path)
        image = im_mat['indian_pines_corrected']
        gt_mat = loadmat(gt_mat_path)
        mask = gt_mat['indian_pines_gt']
     
        image = mean_std_normalize(image)
        self.training = training
        self.sample_percent = sample_percent
        self.batch_size = batch_size   #这里应该类似于定义了一个采样的策略
        super(NewIndianPinesDataset, self).__init__(image, mask, training, np_seed=SEED,
                                                sample_percent=sample_percent,
                                                batch_size=batch_size,batch_size_new = batch_size_new)

    @property
    def num_classes(self):
        return 16
