import torch 
import torchvision
from torchvision import transforms, datasets, models
from path import Path
import pandas as pd
import os 
from sklearn.utils import shuffle
import tqdm
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn
import cv2
import argparse



### pytorch 读取数据的方法
class footstepDataset(torch.utils.data.Dataset):

    def __init__(self,text_file,root_dir,transform,train):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.root_dir = root_dir
        ## transform
        self.transform = transform
        
        ## 读取csv文件
        self.data_csv = pd.read_csv(root_dir + text_file)
        ## 
        if train == True:
            self.data_csv = self.data_csv[self.data_csv['train_and_test'] == 'train']
        if train == False:
            self.data_csv = self.data_csv[self.data_csv['train_and_test'] == 'test']
            self.data_csv = self.data_csv.reset_index()  ## 重置索引          
            
       

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
            ### 读取csv文件中的图片地址
            image = cv2.imread(Path(self.data_csv['path'][idx]))
            ### 对图片进行必要操作
            image = self.transform(image)
            ### 载入身高体重
            target = self.data_csv[['身高(m)','体重(kg)']].values[idx]
            ### 转为tensor
            target = torch.tensor(target).float()
            return image,target


### 指定transform 先转为PIL格式，再将大小变为256*256，转为tensor，再归一化
trans_train= transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256,256)),

        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def metric(pred_list,true_list):
    ###评价
    ### 正确值-预测值/（正确值+1）
    
    ###将模型预测的结果拆分成身高和体重
    height_list = [int(i[0].cpu()) for i in pred_list]
    true_height_list = [int(i[0]) for i in true_list]

    weight_list = [float(i[1].cpu()) for i in pred_list]
    true_weight_list = [float(i[1]) for i in true_list]



    height = []
    weight = []
    #### 进行计算
    for i,j in zip(height_list,true_height_list):
        height.append((abs(i-j)/(i+1)*100))

    for i,j in zip(weight_list,true_weight_list):
        weight.append((abs(i-j)/(i+1)*100))
        
    return np.mean(height),np.mean(weight)

def val(model,val_loader,optimizer,criterion,device,epoch):
#     '''
#     model： 模型结构
#     val_loader： 导入数据的dataloader
#     optimizer： 优化器
#     device： 在cpu还是gpu上进行
#     epoch： 当前运行epoch
#     '''
    model.eval()##进入验证模式
    
    ##存储预测结果
    val_pred_list = []
    val_true_list = []
    ##开始验证
    for image,target in tqdm.tqdm(val_loader):
        ## 将数据送入设备
        image = image.to(device)
        target = target.to(device)
        ## 将数据送入模型
        label = model(image)
        ## 优化器清零
        optimizer.zero_grad()
        ## 记录
        val_pred_list.extend(label)
        val_true_list.extend(target)
        ## 计算loss
        loss = criterion(label,target)
   
    ### 计算指标
    height_metric,weight_metric = metric(val_pred_list,val_true_list)
    print(f'epoch: {epoch}', "val_height_metric:",height_metric,"val_weight_metric:",weight_metric )
    
##训练函数
def train(model,train_loader,test_loader,optimizer,criterion,device,epoch):
#      '''
#     model： 模型结构
#     val_loader： 导入数据的dataloader
#     optimizer： 优化器
#     device： 在cpu还是gpu上进行
#     epoch： 当前运行epoch
#     '''
    ##存储预测结果
    train_pred_list = []
    train_true_list = []

    ## 进入训练模式
    model.train()
    for image,target in tqdm.tqdm(train_loader):
        ## 将数据送入设备
        image = image.to(device)
        target = target.to(device)
        ## 将数据送入模型
        label = model(image)
        ## 优化器清零
        optimizer.zero_grad()
        ### 记录中间变量
        train_pred_list.extend(label)
        train_true_list.extend(target)
        ## 计算loss
        loss = criterion(label,target)
        ## 反向传播
        loss.backward()
         ## 优化器优化
        optimizer.step()
    ### 计算指标
    height_metric,weight_metric = metric(train_pred_list,train_true_list)
    print(f'epoch: {epoch}',"train_height_metric:",height_metric,"train_weight_metric:",weight_metric )
    val(model,test_loader,optimizer,criterion,device,epoch)

    return model ## 返回训练的model


def get_model(arg,device):
    if arg.model == 'vgg16':
        ###vgg16 模型创建过程
        vgg16 = models.vgg16(pretrained=True)##载入预训练模型
        ## 修改最后一层，使输出为2
        vgg16.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=4096, out_features=1280, bias=True),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=1280, out_features=2, bias=True),
        )

        return vgg16.to(device)
    if arg.model == 'resnet34':
        ###resnet34模型创建过程
        resnet34= models.resnet34(pretrained=True)##载入预训练模型
        ### 修改最后一层，使输出为2
        resnet34.fc = nn.Sequential( nn.Linear(in_features=512, out_features=2, bias=True))
        
        return resnet34.to(device)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', nargs='?', type=int, default=50, 
                    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=16, 
                    help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, 
                    help='Learning Rate')
    parser.add_argument('--gpu',nargs='*', type=int, default=1)
    parser.add_argument('--model',nargs='?',type=str,default='vgg16')
    parser.add_argument('--use_pred',nargs='?',type=str,default=None)

    arg = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = arg.l_rate
    n_epoch = arg.n_epoch
    batch_size = arg.batch_size
    model = get_model(arg,device)
    print(arg.model)
    ## 创建该数据集的dataset
    train_dataset = footstepDataset(root_dir = './',text_file = 'label.csv', transform = trans_train, train = True)
    test_dataset = footstepDataset(root_dir = './',text_file = 'label.csv', transform = trans_train, train = False)
    ## 创建该数据集的dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                            batch_size=batch_size,##
                            shuffle=True,## 是否随机打乱
                            num_workers=4
                            )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=batch_size,
                            num_workers=4## 线程数目
                            )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate )
    criterion = nn.MSELoss()

    for epoch in range(n_epoch):
        model = train(model,train_loader,test_loader,optimizer,criterion,device,epoch)
    torch.save(model, f'./models/{n_epoch}_{arg.model}.pth')


def Pred_on_one_pic(model,plist):
# """
#    预测在plist中的图片的身高和体重
# """
    hlist = [] 
    wlist = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in plist:
        temp = cv2.imread(i)
        model.eval()
        temp = trans_train(temp).float()
        temp = temp.unsqueeze_(0)
        temp = temp.to(device)
        model.to(device)
        pred_y = model(temp).cpu()
        pred_result = [float(i) for i in pred_y[0] ]
        hlist.append(pred_result[0])
        wlist.append(pred_result[1])
    return hlist,wlist

def low_than(threshold = 10):
    # """
    # 打印低于threshold的百分比
    # """
    plist = path_label[path_label['train_and_test'] == 'test']['path'].values
    hlist,wlist = Pred_on_one_pic(model,plist)
    h_truelist = path_label[path_label['train_and_test'] == 'test']['身高(m)'].values
    w_truelist = path_label[path_label['train_and_test'] == 'test']['体重(kg)'].values

    metric_h = []
    for i,j in zip(hlist,h_truelist):
        metric_h.append(abs(i-j)*100/j)
    metric_w = []
    for i,j in zip(wlist,w_truelist):
        metric_w.append(abs(i-j)*100/j)
    print(len(metric_h),len([i for i in metric_h if i<threshold]),np.mean(metric_h),len([i for i in metric_h if i< threshold])/len(metric_h))
    print(len(metric_w),len([i for i in metric_w if i<threshold]),np.mean(metric_w),len([i for i in metric_w if i< threshold])/len(metric_w))