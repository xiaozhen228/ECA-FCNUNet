#这一部分主要实现的是迭代训练的过程
from symbol import factor
import torch.nn.functional as F
import torch 
import numpy as np
from module import metric
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import random 
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("训练的仪器是:",device)
def fanzhuan(img,prob=0.6):
    if np.random.rand() < prob:
        d = np.random.randint(-1, 2,1)
        d = d[0]
        img = cv2.flip(img, d)
    else:
        img = img
        d = 28
    return img,d

def xuanzhuan(img,prob=0.6):
    if np.random.rand() < prob:
        factor = np.random.randint(0, 5,1)
        factor = factor[0]
        img = np.rot90(img, factor)     #factor取正数时逆时针旋转，取负数是顺时针旋转
    else:
        img = img
        factor = 28
    return img,factor

def xuanzhuan1(img,prob=0.6):
    if np.random.rand() < prob:
        factor = 2
        img = np.rot90(img, factor)     #factor取正数时逆时针旋转，取负数是顺时针旋转
    else:
        img = img
        factor = 28
    return img,factor

def cutout(img,num_holes=2, max_h_size=4, max_w_size=4,max_k_size=4, fill_value=0, prob=0.1):
    if np.random.rand() < prob:
        print("cutout")
        h = img.shape[0]
        w = img.shape[1]
        k = img.shape[2]
        for _ in range(num_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            z = np.random.randint(k)
            y1 = np.clip(max(0, y - max_h_size // 2), 0, h)  #这个数小于0的变成了0，大于h的变成了h
            y2 = np.clip(max(0, y + max_h_size // 2), 0, h)   #填充的边长为8
            x1 = np.clip(max(0, x - max_w_size // 2), 0, w)
            x2 = np.clip(max(0, x + max_w_size // 2), 0, w)
            z1 = np.clip(max(0, z - max_k_size // 2), 0, k)
            z2 = np.clip(max(0, z + max_k_size // 2), 0, k)
            img[y1: y2, x1: x2, z1 :z2] = fill_value
    else:
        img = img 
    return img

def sp_noise(img,yu=0.01,prob=0.6):
    if np.random.rand() < prob:
        thres = 1 - yu
        rdn = np.random.random(img.shape)
        img[np.where(rdn <yu)] = img.min()    #小于这个概率值的我们使用图像中的最小值来替代，大于这个阈值的用最大值来替代
        img[np.where(rdn>thres)] = img.max()   #其实这两个都是小概率事件
    return img


def gs_noise(img, mean =0,var=0.001,prob =0.6):
    if np.random.rand() < prob:
        print("gs_noise")
        noise = np.random.normal(mean, var ** 0.5, img.shape)
        img = img + noise
    else:
        img = img
    return img

def cube_xuan(img,ground_truth,num_holes=4, max_h_size=25, max_w_size=25,prob = 0.6):
    cube = []
    if np.random.rand() < prob:
        print("进行cube_xuan")
        h = img.shape[0]  #空间维度
        w = img.shape[1]
        
        for i in range(num_holes):
            x1 = np.random.randint(h-max_h_size-1)
            y1 = np.random.randint(w-max_w_size-1)
            x2 = x1 + max_h_size
            y2 = y1 + max_w_size
            
            factor = np.random.randint(0, 5,1)
            factor = factor[0]
            cube.append([x1,y1,x2,y2,factor])
            cube_data = img[x1:x2,y1:y2,:].copy()
            cube_data = np.rot90(cube_data,factor)
            img[x1:x2,y1:y2,:] = cube_data

            ground_truth_cube = np.rot90(ground_truth[x1:x2,y1:y2].copy(),factor)
            ground_truth[x1:x2,y1:y2] = ground_truth_cube

        return img,ground_truth,cube,True
    else:
        return img,ground_truth,cube,False
 
    
class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        #return loss.mean()
        return loss
class Trainer():
    def __init__(self,config,train_dataloader,test_dataloader,model,optimizer,scheduler) :
        self.config = config
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epoch = config["train"]["epoch"]
        
        self.learning_rate = config["learning_rate"]
        self.len_data = len(train_dataloader)
        self.model = model
        self.is_save_model = True
        self.optimizer = optimizer
        self.activate = False
        self.scheduler = scheduler
        self.weight = False
        self.gamma1 = 2
        self.gamma2 = 0
        self.eps = 1e-7
        self.confidence = 0.98
        self.acc_per = 0.6
        
        self.num_unsupervised = config["data"]["train"]["batch_size"] * (config["model"]["num_classes"])
        self.num_unsupervised = self.num_unsupervised * 2
        print("置信度：",self.confidence)
        print("实际中无标签采样中的每类小批量：",config["data"]["train"]["batch_size"])
        print("最多参与训练的无标签样本数量：",self.num_unsupervised)

        if self.config['dataset'] == 'IndianPines':
            self.writer = SummaryWriter('runs/India_experiment_1')
        if self.config['dataset'] == 'pavia':
            self.writer = SummaryWriter('runs/India_experiment_2')
        if self.config['dataset'] == 'salinas':
            self.writer = SummaryWriter('runs/India_experiment_3')
        if self.config['dataset'] == 'houston':
            self.writer = SummaryWriter('runs/India_experiment_4')       

    
    def train_epoch(self,epoch,flag,test_indicator,label_activate=None,list_activate=None):   #这是有两项损失的
        self.model.train()
        train_dataloader = iter(self.train_dataloader)
        for batch_index in range(10):
            self.optimizer.zero_grad()    #清除零梯度 
            zhen_list = []           
            pad_im, pad_mask, train_inds_list,train_inds_list_new = next(train_dataloader)   #这是tensor类型(1,200,160,160),这里都是tensor类型，仿佛就自动转换了
            #pad_mask_yanzhen1 =  pad_mask.clone()
            #pad_im_yanzhen1 = pad_im.clone()
            pad_unim = pad_im.squeeze(0).numpy().copy()
            pad_unim = pad_unim.transpose(1,2,0)

            pad_unmask = pad_mask.squeeze(0).numpy().copy()
            #pad_unim_yanzhen = pad_unim.copy()
            weak_data,id1 = fanzhuan(pad_unim.copy())
            if self.config['dataset'] == 'IndianPines':
                weak_data,id2 = xuanzhuan(weak_data)
            elif self.config['dataset'] == 'pavia':
                weak_data,id2 = xuanzhuan(weak_data)
            elif self.config['dataset'] == 'salinas':
                weak_data,id2 = xuanzhuan(weak_data)
            else:
                weak_data,id2 = xuanzhuan(weak_data)
            strong_data,pad_mask_st,cube_list_st,qu_st = cube_xuan(pad_unim.copy(),pad_unmask.copy(),1,100,100,0.4)
            #print(cube_list)
            
            strong_data,id3 = fanzhuan(strong_data)
            if self.config['dataset'] == 'IndianPines':
                strong_data,id4 =xuanzhuan(strong_data)
            elif self.config['dataset'] == 'pavia':
                strong_data,id4 = xuanzhuan(strong_data)
            elif self.config['dataset'] == 'salinas':
                strong_data,id4 = xuanzhuan(strong_data)
            else:
                strong_data,id4 = xuanzhuan(strong_data)
            #strong_data = cutout(strong_data)
            strong_data = gs_noise(strong_data)
            zhen_list = [id1,id2,id3,id4]
            weak_data = weak_data.transpose(2,0,1).copy()
            strong_data = strong_data.transpose(2,0,1).copy()

            weak_data = torch.from_numpy(weak_data).unsqueeze(0).to(torch.float32)
            weak_data = weak_data.to(device)
            strong_data = torch.from_numpy(strong_data).unsqueeze(0).to(torch.float32)
            strong_data = strong_data.to(device)
            pad_im=pad_im.to(device)
            pad_mask = pad_mask.to(device)
            pad_mask_st =torch.from_numpy(pad_mask_st).unsqueeze(0)
            pad_mask_st = pad_mask_st.to(device)
            train_inds_list = train_inds_list.to(device)
            train_inds_list_new = train_inds_list_new.to(device)
            if flag== 0:
                yuan_data,yuan_mask,cube_list,qu = cube_xuan(pad_unim.copy(),pad_unmask.copy())
                yuan_data = yuan_data.transpose(2,0,1).copy()
                yuan_data = torch.from_numpy(yuan_data).unsqueeze(0).to(torch.float32).to(device)
                yuan_mask = torch.from_numpy(yuan_mask).unsqueeze(0).to(device)
                logit1 = self.model(yuan_data,pad_mask,train_inds_list)     #这是没有经过softmax的(1,16,160,160)
                train_loss1 = self.loss1(logit1,pad_mask,train_inds_list,yuan_mask,cube_list,qu)
                if self.weight:
                    hope = 1
                    logit2 = self.model(weak_data,pad_mask,train_inds_list_new)
                    logit3 = self.model(strong_data,pad_mask_st,train_inds_list_new)
                    train_loss2 = self.loss2(logit2,logit3,test_indicator.copy(),zhen_list,cube_list_st,qu_st,pad_mask_st,pad_mask)
                    train_loss = train_loss1 +train_loss2
                    self.writer.add_scalar('unsup_loss', train_loss2.item(), global_step=epoch*10+batch_index)
                else:
                    hope = 0
                    train_loss2 = 0*train_loss1
                    train_loss = train_loss1
                
                self.writer.add_scalar('sup_loss', train_loss1.item(), global_step=epoch*10+batch_index)
                #self.writer.add_scalar('unsup_loss', train_loss2.item(), global_step=epoch*10+batch_index)
                self.writer.add_scalar('total_loss', train_loss.item(), global_step=epoch*10+batch_index)
                print(train_loss1.item(),train_loss2.item(),train_loss.item())
                train_loss.backward()      #这里才是真正的反向传播部分
                self.optimizer.step()
            else:
                logit1 = self.model(pad_im,pad_mask,train_inds_list) 
                logit2 = self.model(weak_data,pad_mask,train_inds_list_new)
                logit3 = self.model(strong_data,pad_mask,train_inds_list_new)
                
                train_loss1 = self.loss1(logit1,pad_mask,train_inds_list)
                train_loss2 = self.loss2(logit2,logit3,train_inds_list_new,zhen_list,epoch)
                train_loss3 = self.loss3(logit1,label_activate,list_activate[batch_index])
                train_loss = train_loss1 + train_loss3 + train_loss2
                self.writer.add_scalar('activate_loss', train_loss3.item(), global_step=epoch*10+batch_index)
                self.writer.add_scalar('total_loss', train_loss.item(), global_step=epoch*10+batch_index)
                print(train_loss1.item(),train_loss3.item(),train_loss.item())
                train_loss.backward()
                self.optimizer.step()
        return logit1,train_loss1

    def train(self):
        print("开始训练，共%d 个周期"%self.epoch)
        aa_m = 0
        tempp =1
        r=random.random
        random.seed(100)
        test_indicator = np.load('test_indicator.npy')
        storage = torch.zeros(10,test_indicator.shape[0]*test_indicator.shape[1])
        test_indicator1 = torch.from_numpy(test_indicator.copy()).to(device)
        f = open('./zhibiao.txt','a+')
        f.write("==================================================================="+"\n")
        f1 = open('./zhibiao_new.txt','a+')
        f1.write("==================================================================="+"\n")        
        for epoch in range(self.epoch):
            print("=========== epoch=%d ==========="%epoch)
            print(self.optimizer.state_dict()['param_groups'][0]['lr'])
            if self.activate:
                if (epoch %10 ==0):
                    label_activate = torch.argmax(data,dim=0) + 1
                    list_activate = []
                    for i in range(10):
                        active_inds = torch.zeros_like(test_indicator1).flatten()
                        random.shuffle(idxs,random=r)
                        cd = int((idxs.shape[0])*0.2)
                        fetch_inds = idxs[:cd]
                        active_inds[fetch_inds] = 1
                        list_activate.append(active_inds.reshape(test_indicator.shape)) 
                        
                if (tempp ==1):
                    label_activate = torch.argmax(data,dim=0) + 1
                    list_activate = []
                    for i in range(10):
                        active_inds = torch.zeros_like(test_indicator1).flatten()
                        random.shuffle(idxs,random=r)
                        cd = int((idxs.shape[0])*0.2)
                        fetch_inds = idxs[:cd]
                        active_inds[fetch_inds] = 1
                        list_activate.append(active_inds.reshape(test_indicator.shape)) 
                        tempp = 0
                data,loss  =  self.train_epoch(epoch,label_activate=label_activate,list_activate=list_activate,flag=1)  #因为我想的是在前期进行一致性正则的时候我们用这个方法
            else:
                
                data,loss = self.train_epoch(epoch,flag=0,test_indicator = test_indicator.copy())
            self.scheduler.step(loss)
            
            data = torch.softmax(data,dim=1).squeeze()
            
            storage[epoch%10,:] = (torch.argmax(data,dim=0) + 1).flatten()   #这里存储的是预测的标签
            
                
            if (epoch %10 ==0) and (epoch>=10) :
                temp = (storage[0,:] == storage[1,:])
                for i in range(2,10):
                    temp = temp*(storage[i,:] == storage[0,:]) #所有和第一行是公共的元素就是公共元素
                test_indicator1 = test_indicator1.flatten()
                temp = temp.to(device)
                pre_value  = torch.max(data,dim=0)[0].flatten()
                index = torch.where((test_indicator1==1) * temp *(pre_value > 0.95))  #这些标签已经符合了，十次每一次都是一样的而且是测试集里的
               
                idxs = index[0]
                print(idxs.shape[0])
                if idxs.shape[0] > 8000000:
                    self.activate = True
                    print("有希望")
                else:
                    self.activate =False
                    f.write(str(idxs.shape[0])+' ')
                    

            
                oa,aa,kappa,acc_per_class = self.fcn_evaluate()
                if (acc_per_class>self.acc_per).all():    #统计测试集中连续10次都预测结果不变而且置信度大于0.95的样本数量
                    self.weight = True
                else:
                    self.weight = False
                print(self.weight)
                f.write(str(self.weight)+' ')
                f.write(str(round(oa,4))+' ')
                f.write(str(round(aa,4))+' ')
                f.write(str(round(kappa,4))+' ')
                f.write(str(np.around(acc_per_class.cpu().numpy(), decimals=4))+'\n')
                f.flush()
                if self.is_save_model:
                    if aa>aa_m:
                        aa_m = aa
                        state = {'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'epoch':epoch}
                        if self.config['dataset'] == 'IndianPines':
                            torch.save(state, "./log_dir_india/model_new.pth")
                        elif self.config['dataset'] == 'pavia':
                            torch.save(state, "./log_dir_pavia/model_new.pth")
                        elif self.config['dataset'] == 'salinas':
                            torch.save(state, "./log_dir_salinas/model_new.pth")
                        else:
                            torch.save(state, "./log_dir_houston/model_new.pth")
                oa,aa,kappa,acc_per_class = self.fcn_evaluate_train()
                f.write(str(self.weight)+' ')
                f1.write(str(round(oa,4))+' ')
                f1.write(str(round(aa,4))+' ')
                f1.write(str(round(kappa,4))+' ')
                f1.write(str(np.around(acc_per_class.cpu().numpy(), decimals=4))+'\n')
                f1.flush()                
        f.write("==================================================================="+"\n")
        f.close()
        f1.write("==================================================================="+"\n")
        f1.close()
                
            
    def loss1(self, x, y,weight,yuan_mask,cube_list,qu):  
        x = x.squeeze(0)
        if qu:
            print("有监督之前看二者是否相等",yuan_mask.equal(y))
            cube_list = cube_list[::-1]
            for i in range(len(cube_list)):
                cube_index = cube_list[i]
                x1 = cube_index[0]
                y1 = cube_index[1]
                x2 = cube_index[2]
                y2 = cube_index[3]
                factor = cube_index[4]
                #print(logit2.shape)
                #print(logit2[:,x1:x2,y1:y2].shape)
                logit3 = torch.rot90(x[:,x1:x2,y1:y2],-1*int(factor),dims=[1,2])
                logit4 = torch.rot90(yuan_mask[:,x1:x2,y1:y2],-1*int(factor),dims=[1,2])
                #print(logit3.shape)
                x[:,x1:x2,y1:y2] = logit3
                yuan_mask[:,x1:x2,y1:y2] = logit4
            x= x.unsqueeze(0)
            print("有监督之后看二者是否相等",yuan_mask.equal(y))
        else:
            x = x.unsqueeze(0)
        
        losses = F.cross_entropy(x, y.long() - 1, ignore_index=-1, reduction='none')
        #p = torch.exp(-losses)
        #losses = (1 - p) ** self.gamma1 * losses
        losses2 = losses.mul_(weight).sum() / weight.sum()  #F.cross_entropy不经过softmax处理
        return losses2

    def loss2(self,logit1,logit2,weight,zhen,cube_list,qu,st_mask,mask):  #我需要把logit1变为硬标签,logit2变为预测值
        logit1 = logit1.squeeze(0)
        weight = torch.from_numpy(weight).to(device)
        print(zhen)
        if zhen[1] == 28:
            logit1 = logit1
        else:
            logit1 = torch.rot90(logit1,-1*int(zhen[1]),dims=[1,2])   #逆时针旋转
        if zhen[0] == -1:
            logit1 = torch.flip(logit1,dims=[1,2])
        if zhen[0] == 0:
            logit1 = torch.flip(logit1,[1])
        if zhen[0] == 1:
            logit1 = torch.flip(logit1,[2])
        if zhen[0] == 28:
            logit1 = logit1
        

        logit2 = logit2.squeeze(0)
        if zhen[3] == 28:
            logit2 = logit2
        else:
            logit2 = torch.rot90(logit2,-1*int(zhen[3]),dims=[1,2])   #逆时针旋转
        if zhen[2] == -1:
            logit2 = torch.flip(logit2,dims=[1,2])
        if zhen[2] == 0:
            logit2 = torch.flip(logit2,[1])  #垂直翻转
        if zhen[2] == 1:
            logit2 = torch.flip(logit2,[2])   #水平翻转
        if zhen[2] == 28:
            logit2 = logit2
        if qu:
            print("无监督之前看二者是否相等",st_mask.equal(mask))
            cube_list = cube_list[::-1]
            for i in range(len(cube_list)):
                cube_index = cube_list[i]
                x1 = cube_index[0]
                y1 = cube_index[1]
                x2 = cube_index[2]
                y2 = cube_index[3]
                factor = cube_index[4]
                #print(logit2.shape)
                #print(logit2[:,x1:x2,y1:y2].shape)
                logit3 = torch.rot90(logit2[:,x1:x2,y1:y2],-1*int(factor),dims=[1,2])
                logit4 =  torch.rot90(st_mask[:,x1:x2,y1:y2],-1*int(factor),dims=[1,2])
                #print(logit3.shape)
                logit2[:,x1:x2,y1:y2] = logit3
                st_mask[:,x1:x2,y1:y2] = logit4
            logit2 = logit2.unsqueeze(0)
            print("无监督之后看二者是否相等",st_mask.equal(mask))
        else:
            logit2 = logit2.unsqueeze(0)

        r=random.random
        random.seed(228)
        logit1 = torch.softmax(logit1,dim=0)
        label = torch.argmax(logit1,dim=0) +1   #二维
        pre_value = torch.max(logit1,dim=0)[0]

        pre = torch.where((pre_value <= self.confidence).mul((weight)==1))   #不能参与运算的无标签样本
        #print("没参加:",pre[0].shape[0])
        weight[pre[0],pre[1]] = 0
        weight1 = weight.clone().flatten()
        indxx = torch.where(weight1 ==1)[0]
        xiao = [i for i in range(len(indxx))]
        random.shuffle(xiao,random=r)   
        #random.shuffle(indxx,random=r)  #这样写是错误的，真他吗气人
        indxx = indxx[xiao]
        if torch.sum(weight) <= self.num_unsupervised:
            weight1 = weight1
        else:
            print("原无标签：",torch.sum(weight1).item())
            weight1[indxx[self.num_unsupervised:]] = 0     #其实这个weight已经改变了。这里经常有问题
            print("新标签：",torch.sum(weight1).item())
        weight = weight1.reshape(weight.shape)
        #print("无标签：",torch.sum(weight).item())
        weight = weight.unsqueeze(0)
        label = label.unsqueeze(0)


        losses = F.cross_entropy(logit2, label.long() - 1, ignore_index=-1, reduction='none')  #记住这里是标签从0开始计算损失，数据集中的0是地面
        #p = torch.exp(-losses)
        #losses = (1 - p) ** self.gamma2 * losses
        losses2 = losses.mul_(weight).sum() / (weight.sum()+self.eps)  #F.cross_entropy不经过softmax处理
        return losses2


    def loss3(self,logit,label_activate,list_activate):
        label_activate = label_activate.unsqueeze(0).to(device)
        list_activate = list_activate.unsqueeze(0).to(device)
        #print(list_activate.shape)
        beta = 0.9999  #x就是那个没有经过softmax的预测值
        if self.config['dataset'] == "pavia":
            cls_num_list = [6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
        elif self.config['dataset'] =='IndianPines':
            cls_num_list = [46, 1428, 830, 237, 483, 730, 28, 478, 20, 927, 2455, 593, 205, 1265, 386, 93]
        elif self.config['dataset'] == "salinas":
            cls_num_list = [2009,3726,1976,1394,2678,3959,3579,11271,6203,3278,1068,1927,916,1070,7268,1807]
        elif self.config['dataset'] == "houston":
            cls_num_list = [1251,1254,697,1244,1242,325,1268,1244,1252,1227,1235,1233,469,428,660]
        else:
            print("no cls_num_list")
        effective_num = 1.0 - np.power(beta, cls_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
        per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)
        losses = F.cross_entropy(logit, label_activate.long() - 1, ignore_index=-1, reduction='none', weight=per_cls_weights)
        #print(losses.shape)
        losses3 = losses.mul_(list_activate).sum() / list_activate.sum()  #F.cross_entropy不经过softmax处理
        return losses3


        
    def th_confusion_matrix(self,y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
        """

        Args:
            y_true: 1-D tensor of shape [n_samples], label value starts from 1 rather than 0
            y_pred: 1-D tensor of shape [n_samples]
            num_classes: scalar
        Returns:

        """
        size = [num_classes + 1, num_classes + 1] if num_classes is not None else None
        y_true = y_true.float()
        y_pred = y_pred.float()
        if size is None:
            cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred))
        else:
            cm = torch.sparse_coo_tensor(indices=torch.stack([y_true, y_pred], dim=0), values=torch.ones_like(y_pred),
                                        size=size)

        return cm.to_dense()[1:, 1:] if cm.size(0) > 2 else cm.to_dense()

    def th_cohen_kappa_score(self,y_true: torch.Tensor, y_pred: torch.Tensor, num_classes=None):
        cm_th = self.th_confusion_matrix(y_true, y_pred, num_classes)
        cm_th = cm_th.float()
        n_classes = cm_th.size(0)
        sum0 = cm_th.sum(dim=0)
        sum1 = cm_th.sum(dim=1)
        expected = torch.ger(sum0, sum1) / torch.sum(sum0)
        w_mat = torch.ones([n_classes, n_classes], dtype=torch.float32).to(device)
        w_mat.view(-1)[:: n_classes + 1] = 0.
        k = torch.sum(w_mat * cm_th) / torch.sum(w_mat * expected)
        return 1. - k




    def fcn_evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (im, mask, w,w1) in enumerate(self.test_dataloader):
                im = im.to(device)
                mask = mask.to(device)
                w = w.to(device)
                y_pred = self.model(im).squeeze()  #tensor
                #torch.cuda.synchronize()
                y_pred = y_pred.argmax(dim=0).to(device) + 1   #这里加了1，所以应该第0类没用，是地面
                w.unsqueeze_(dim=0)
                w = w.byte()
                mask = torch.masked_select(mask.view(-1), w.view(-1))  #根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量
                y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))  #这是展开成为一维的意思
                oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
                aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                    self.config["model"]["num_classes"],
                                                                    return_accuracys=True)
                kappa = self.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self.config["model"]["num_classes"])
                print("测试集指标")
                print(oa.item(),aa.item(),kappa.item())
                print(acc_per_class)
        return oa.item(),aa.item(),kappa.item(),acc_per_class

    def fcn_evaluate_train(self):
        self.model.eval()
        with torch.no_grad():
            for idx, (im, mask, w,w1) in enumerate(self.test_dataloader):
                im = im.to(device)
                mask = mask.to(device)
                w1 = w1.to(device)
                y_pred = self.model(im).squeeze()  #tensor
                #torch.cuda.synchronize()
                y_pred = y_pred.argmax(dim=0).to(device) + 1   #这里加了1，所以应该第0类没用，是地面
                w1.unsqueeze_(dim=0)
                w1 = w1.byte()
                mask = torch.masked_select(mask.view(-1), w1.view(-1))  #根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量
                y_pred = torch.masked_select(y_pred.view(-1), w1.view(-1))  #这是展开成为一维的意思
                oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
                aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                    self.config["model"]["num_classes"],
                                                                    return_accuracys=True)
                print("训练集指标")
                kappa = self.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self.config["model"]["num_classes"])
                print(oa.item(),aa.item(),kappa.item())
                print(acc_per_class)
        return oa.item(),aa.item(),kappa.item(),acc_per_class
    def fcn_evaluate_map(self):
        
        self.model.eval()
        
        y_all_list = []
        y_all_gt = []
        with torch.no_grad():
            for idx, (im, mask, w,w1) in enumerate(self.test_dataloader):
                im = im.to(device)
                mask = mask.to(device)
                w = w.to(device)
                y_pred = self.model(im).squeeze()  #tensor
                #torch.cuda.synchronize()
                np.save("y_pred.npy",y_pred.cpu().detach().numpy())
                y_pred = y_pred.argmax(dim=0).to(device) + 1
                w.unsqueeze_(dim=0)
                if self.config['dataset'] == 'IndianPines':
                    y_out = y_pred[0:145, 0:145]
                elif self.config['dataset'] == 'pavia':
                    y_out = y_pred[0:610, 0:340]
                elif self.config['dataset'] == 'salinas':
                    y_out = y_pred[0:512, 0:217]
                else:
                    y_out = y_pred[0:349, 0:1905]
               
                w = w.byte()

                mask = torch.masked_select(mask.view(-1), w.view(-1))  #根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量

                y_pred = torch.masked_select(y_pred.view(-1), w.view(-1))  #这是展开成为一维的意思
                if self.config['dataset'] == 'IndianPines':
                    gt_mat = loadmat('./IndianPines/Indian_pines_gt.mat')
                    gt_mask = gt_mat['indian_pines_gt']
                elif self.config['dataset'] == 'pavia':
                    gt_mat = loadmat("./pavia/PaviaU_gt.mat")
                    gt_mask = gt_mat['paviaU_gt']
                elif self.config['dataset'] == 'salinas':
                    gt_mat = loadmat("./salinas/Salinas_gt.mat")
                    gt_mask = gt_mat['salinas_gt']
                else:
                    gt_mat = loadmat("./HoustonU/Houston_gt.mat")
                    gt_mask = gt_mat['Houston_gt']

                gt = gt_mask.flatten()
                x_label = np.zeros(gt.shape)
                y_label = np.zeros(gt.shape)
                for i in range(len(gt)):    #对真实的ground_turth进行处理
                    if gt[i] == 0:
                        gt[i] = 17
                        x_label[i] = 16     #这里似乎没有用

                gt = gt[:] - 1    #这样就把真实的黑色变成了16号
                y_out = y_out.flatten()
                for i in range(len(y_out)):  #这样操作就没有考虑预测为地面的情况，因为分类的时候就只有16类，黑色一定会被错误分某一类
                    if y_out[i] == 0:
                        y_out[i] = 17
                        y_label[i] = 16
                y_out = y_out[:] - 1
                x = np.ravel(y_out.cpu())
                y_list = list_to_colormap(x)   #将预测的进行一个颜色映射
                y_gt = list_to_colormap(gt)    #将真实真实的标签颜色映射
                y_all_list.append(y_list)
                y_all_gt.append(y_gt)
                y_re = np.reshape(y_list, (gt_mask.shape[0], gt_mask.shape[1], 3))
                gt_re = np.reshape(y_gt, (gt_mask.shape[0], gt_mask.shape[1], 3))
                if self.config['dataset'] == 'IndianPines':
                    classification_map(y_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'indian_pines.png')  #保存图像的过程
                    classification_map(gt_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'indian_pines_gt.png')  #保存图像
                elif self.config['dataset'] == 'pavia':
                    classification_map(y_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'pavia.png')  #保存图像的过程
                    classification_map(gt_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'pavia_gt.png')  #保存图像     
                elif self.config['dataset'] == 'salinas':
                    classification_map(y_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'salinas.png')  #保存图像的过程
                    classification_map(gt_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'salinas_gt.png')  #保存图像  
                else:
                    classification_map(y_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'houston.png')  #保存图像的过程
                    classification_map(gt_re, gt_mask, 300,
                                './classification_maps/' + str(0.02) + '_' + 'houston_gt.png')  #保存图像                                       
                oa = metric.th_overall_accuracy_score(mask.view(-1), y_pred.view(-1))
                aa, acc_per_class = metric.th_average_accuracy_score(mask.view(-1), y_pred.view(-1),
                                                                    self.config["model"]["num_classes"],
                                                                    return_accuracys=True)
                kappa = self.th_cohen_kappa_score(mask.view(-1), y_pred.view(-1), self.config["model"]["num_classes"])
                print(oa.item(),aa.item(),kappa.item())
                print(acc_per_class)
            


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.    #黑色的
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.    #黑色的
    return y


