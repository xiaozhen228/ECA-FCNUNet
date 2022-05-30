from torch.utils.data.dataloader import DataLoader

from data.pavia import NewPaviaDataset
from data.base import MinibatchSampler
from data.salinas import NewSalinasDataset
from data.indianpine import NewIndianPinesDataset
from data.HOS import NewHOSDataset

class NewPaviaLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v          #实例的属性

        dataset = NewPaviaDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                  self.sample_percent, self.batch_size,self.batch_size_new)  
        sampler = MinibatchSampler(dataset)
        self.data_num = len(dataset) 
        super(NewPaviaLoader, self).__init__(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=sampler,
                                             batch_sampler=None,
                                             num_workers=self.num_workers,
                                             pin_memory=True,
                                             drop_last=False,
                                             timeout=0,
                                             worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )



class NewSalinasLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewSalinasDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size,self.batch_size_new)   #这里才是你真正的定义，随意采样比例还是由配置所决定的
        self.data_num = len(dataset)
        sampler = MinibatchSampler(dataset)   #这个和shuffle是互斥的，但是有用极了
        #我感觉这里使用的是pytorch自己的dataloader
        super(NewSalinasLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )


class NewIndianPinesLoader(DataLoader):   #迭代器这里设计有问题
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():   #这里已经把config中的属性添加了进去
            self.__dict__[k] = v        

        dataset = NewIndianPinesDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size,self.batch_size_new)   #NewIndianPinesDataset其实是一个定义好的类
        sampler = MinibatchSampler(dataset)   #应该是采样方法  #构建这个数据集时调用了FullImageDataset，执行了一些save
        self.data_num = len(dataset)    #数据集的长度
        super(NewIndianPinesLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)   #这是为了继承pytorch中自定义的dataloader
#这里的batchsize设计的是1
    def set_defalut(self):    #我觉得可能存在某种机制先执行这个函数
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )

class NewHOSLoader(DataLoader):
    def __init__(self, config):
        self.config = dict()
        self.set_defalut()
        self.config.update(config)
        for k, v in self.config.items():
            self.__dict__[k] = v

        dataset = NewHOSDataset(self.image_mat_path, self.gt_mat_path, self.training,
                                    self.sample_percent, self.batch_size,self.batch_size_new)
        sampler = MinibatchSampler(dataset)
        self.data_num = len(dataset)
        super(NewHOSLoader, self).__init__(dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               sampler=sampler,
                                               batch_sampler=None,
                                               num_workers=self.num_workers,
                                               pin_memory=True,
                                               drop_last=False,
                                               timeout=0,
                                               worker_init_fn=None)

    def set_defalut(self):    #我觉得可能存在某种机制先执行这个函数
        self.config.update(dict(
            num_workers=0,
            image_mat_path='',
            gt_mat_path='',
            training=True,
            sample_percent=0.01,
            batch_size=10)
        )

