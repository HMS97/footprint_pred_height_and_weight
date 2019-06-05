import numpy as np
from path import Path
import pandas as pd


### 读取xlsx文件
data = pd.read_excel('ZY122.xlsx',error_bad_lines=False)
### 身高转化为厘米
data['身高(m)'] = data['身高(m)']*100

path_label = pd.DataFrame()
image_path = Path('压力图ZY2未删减')
path_list = []
### 遍历图像获取图像的路径
for i in image_path.walkdirs():
    for j in i.walkfiles():
        path_list.append(str(j))
        
### 利用每个图像文件的身份证号做索引
path_label['path'] = path_list
path_class = []
for i in  path_label.path.str.split('/'):
    path_class.append(i[1])
path_label['身份证'] = path_class

##连接两个csv文件
path_label = path_label.merge(data,on='身份证')
### 写如具体路径
path_label['path'] = os.getcwd()+ '/' +  path_label['path']
### 随机打散
path_label = shuffle(path_label)
### 数据集减去验证集
train_and_test = ['train'] *(len(path_label) - len(path_label)//5) 
### 指定训练集和验证集
train_and_test.extend(['test']*(len(path_label)//5 ))
path_label['train_and_test'] = train_and_test

##保存到label，csv中
path_label.to_csv('label.csv',index = False)
