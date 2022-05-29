#这一部分主要实现的加入一些自己的参数之类的，然后把datalaoder和dataset设计到main()函数里面
from symbol import factor
import torch
from module import FPGA, SSDGL
from torch.utils.data.dataloader import DataLoader
from scipy.io import loadmat
import argparse
import json
import torch.optim as optim
from data.dataloader import NewIndianPinesLoader,NewPaviaLoader,NewSalinasLoader,NewHOSLoader
from trainer import Trainer
from module.SSDGL import SSDGL 
from module.FPGA import FreeNet 
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用的设备为：",device)
def main(config):
    '''
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    '''
    print(config['dataset'])
    if config['dataset'] == 'IndianPines':
        #train_dset = NewIndianPinesDataset(config["data"]["train"]["image_mat_path"],config["data"]["train"]["gt_mat_path"])
        train_dataloader = NewIndianPinesLoader(config["data"]["train"])
        test_dataloader = NewIndianPinesLoader(config["data"]["test"])
        print(train_dataloader.data_num) #10  这是由随机批量数来决定的
        print(test_dataloader.data_num)  #1
    if config['dataset'] == 'pavia':
        train_dataloader = NewPaviaLoader(config["data"]["train"])
        test_dataloader = NewPaviaLoader(config["data"]["test"])
        print(train_dataloader.data_num) #10  这是由随机批量数来决定的
        print(test_dataloader.data_num)  #1
    if config['dataset'] == 'salinas':
        train_dataloader = NewSalinasLoader(config["data"]["train"])
        test_dataloader = NewSalinasLoader(config["data"]["test"])
        print(train_dataloader.data_num) #10  这是由随机批量数来决定的
        print(test_dataloader.data_num)  #1  
    if config['dataset'] == 'houston':
        train_dataloader = NewHOSLoader(config["data"]["train"])
        test_dataloader = NewHOSLoader(config["data"]["test"])
        print(train_dataloader.data_num) #10  这是由随机批量数来决定的
        print(test_dataloader.data_num)  #1 
    #model = SSDGL(config).to(device)
    model = FreeNet(config).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001,momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor = 0.6,patience=50)
    load_model = False
    if config['dataset'] == 'IndianPines':
        if load_model:
            checkpoint = torch.load("./log_dir_india/model_new.pth")
            model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            print('无保存模型，将从头开始训练！')
    elif config['dataset'] == 'pavia':
        if load_model:
            checkpoint = torch.load("./log_dir_pavia/model_new.pth")
            model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            print('无保存模型，将从头开始训练！')
    elif config['dataset'] == 'salinas':
        if load_model:
            checkpoint = torch.load("./log_dir_salinas/model_new.pth")
            model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))  
        else:
            print('无保存模型，将从头开始训练！')
    else:
        if load_model:
            checkpoint = torch.load("./log_dir_houston/model_new.pth")
            model.load_state_dict(checkpoint['model'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch)) 
        else:
            print('无保存模型，将从头开始训练！') 
    trainer = Trainer(config,train_dataloader,test_dataloader,model,optimizer,scheduler)
    trainer.train()
    trainer.fcn_evaluate_map()



if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    
    #parser.add_argument('-c', '--config', default='configs/config_india.json',type=str,help='Path to the config file')
    parser.add_argument('-c', '--config', default='configs/config_pavia.json',type=str,help='Path to the config file')
    #parser.add_argument('-c', '--config', default='configs/config_salinas.json',type=str,help='Path to the config file')
    #parser.add_argument('-c', '--config', default='configs/config_houston.json',type=str,help='Path to the config file')

    parser.add_argument('-n', '--minibatch', default=4,type=int,help='the number of mninbatch')
    
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--cpu', action='store_true', default=False, help='')
    parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
    args = parser.parse_args()

    config = json.load(open(args.config))    #其实就是将命令行参数的那个地址字符串赋予了--config中的config变量
    config["data"]["train"]["batch_size"] = args.minibatch
    torch.backends.cudnn.benchmark = True   #加速模型的训练
    #print(config)  #这样解析出来的就都是字典了类型了，其实也挺好用
    #print(config["data"]["train"]["image_mat_path"],config["data"]["train"]["gt_mat_path"])
    main(config)

    


    
    
    
    
    
