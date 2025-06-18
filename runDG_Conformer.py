import os
gpus = [0]
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ["CUDA_VISIBLE_DEVICES"] = '1' 
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.autograd as autograd
from torchvision.models import vgg19
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import Tensor

from PIL import Image
import torchvision.transforms as transforms
from sklearn.decomposition import PCA

from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
import scipy.signal as signal
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter
from sklearn import preprocessing
from sklearn import manifold
from sklearn.model_selection import train_test_split
from modelDG_Conformer import Conformer

from torchvision import transforms
from sklearn.cross_decomposition import CCA
from scipy.signal import cheb1ord, filtfilt, cheby1
from sklearn.manifold import TSNE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

from thop import profile
from stablenet_utils.stablenet_reweighting import weight_learner
from stablenet_utils.config_DG_Conformer import parser
import tsaug

def data_standardization(Raw):
    b = Raw.shape[0]
    after = np.zeros_like(Raw)
    for i_stand in range(b):
        after[i_stand] = preprocessing.scale(Raw[i_stand], axis=0)
    return after

fs = 250
bp_low = 6
bp_high = 81
Wn1 = bp_low * 2 / fs
Wn2 = bp_high * 2 / fs
b, a = signal.butter(6, [Wn1, Wn2], 'bandpass')      # 配置滤波器6阶  # Wn=2*截止频率/采样频率

w_notch = 2 * 50 / fs        
b_notch, a_notch = signal.iirnotch(w_notch, 300)     # Q=300表示质量因数,越大滤波的范围越窄(目标频率周围)
def bandpassfilter(Raw):
    filtedData = np.zeros_like(Raw)
    for i in range(Raw.shape[0]):
        for j in range(Raw.shape[1]):
            detrended = signal.detrend(Raw[i, j, :], type='linear')
            notched = signal.filtfilt(b_notch, a_notch, detrended)
            filtedData[i, j, :] = signal.filtfilt(b, a, notched)          
    return filtedData


def get_source_SSVEPdata(targetS, datafolder):
    window_length = 200                              # 数据点窗口长度,其影响模型中间特征图大小, 所以改变窗口长度的时候还需改动模型(classification的linear等处)
    channel_zhenqu = [47,53,54,55,56,57,60,61,62]          

    phase_path = r'/home/ljw/dataset_opensrc/Tsinghua BCI Lab Benchmark Dataset/Freq_Phase.mat'
    phase_info = scipy.io.loadmat(phase_path)
    phase_info = phase_info['phases']

    S_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16',
              'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30',
              'S31', 'S32', 'S33', 'S34', 'S35']
    target_input = []
    target_label = []
    train_input = []
    train_label = []

    # target subject
    target_S = targetS
    target_Spath = os.path.join(datafolder, '{0}.mat'.format(target_S))
    target_Sdata = scipy.io.loadmat(target_Spath)
    target_Sdata = target_Sdata['data']
    target_Sdata = target_Sdata[channel_zhenqu, 160:160 + window_length, :, :]  # 原始Benchmark数据集，每个被试的数据shape 64 1500 40 6

    for num_target in range(40):
        for num_block in range(6):    
            target_input.append((target_Sdata[:, :, num_target, num_block]))
            target_label.append(num_target)
 

    target_input = np.array(target_input)
    target_label = np.array(target_label)
    target_input = bandpassfilter(target_input)
    # target_input = data_standardization(target_input)
    target_input = target_input[:, np.newaxis, :, :]                            # 插入新维度后num_data * feat_channel * eeg_channel * time_point
    # print(target_label)                                                       # 每种刺激频率的6个试次连在一起

    # train subjects (other 34 except target)
    S_list.remove(target_S)                                                     
    train_Ss = S_list
    print('train_Ss:', train_Ss)
    for num_subject in range(len(train_Ss)):  
        train_Sspath = os.path.join(datafolder, '{0}.mat'.format(train_Ss[num_subject]))
        train_Ssdata = scipy.io.loadmat(train_Sspath)
        train_Ssdata = train_Ssdata['data']
        train_Ssdata = train_Ssdata[channel_zhenqu, 160:160 + window_length, :, :]  


        for num_target in range(40):
            for num_block in range(6):
                train_input.append((train_Ssdata[:, :, num_target, num_block]))
                train_label.append(num_target)
 
    train_input = np.array(train_input)
    train_label = np.array(train_label)
    train_input = bandpassfilter(train_input)
    # train_input = data_standardization(train_input)
    train_input = train_input[:, np.newaxis, :, :]

    return train_input, train_label, target_input, target_label


def intermediate_visualize(data, label):
    data = data.reshape((data.shape[0], -1))
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(data)
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    for i in range(40):
        plt.scatter(X_tsne[label == i, 0], X_tsne[label == i, 1], label='class {0}'.format(i))
    plt.show()
    return None




def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):      # 多核MMD的高斯核创建
    n_samples = int(source.shape[0]) + int(target.shape[0])  # 求矩阵的行数
    total = torch.cat([source, target], dim=0)               # 将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):          # 多核MMD

    sourcebatch_size = int(source.shape[0])
    targetbatch_size = int(target.shape[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    # 将核矩阵分成4部分
    XX = kernels[:sourcebatch_size, :sourcebatch_size]
    YY = kernels[sourcebatch_size:, sourcebatch_size:]
    XY = kernels[:sourcebatch_size, targetbatch_size:]
    YX = kernels[targetbatch_size:, :sourcebatch_size]
    
    #XX = torch.div(XX, sourcebatch_size * sourcebatch_size).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
    #XY = torch.div(XY, -sourcebatch_size * targetbatch_size).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target
    #YX = torch.div(YX, -targetbatch_size * sourcebatch_size).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
    #YY = torch.div(YY, targetbatch_size * targetbatch_size).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
    	
    loss = torch.mean(XX + YY - XY -YX)     # 此运算是按照X和Y形状一样的情况计算

    return loss 


# 困难样本挖掘，选择损失大的样本进行学习而忽略小样本
class OhemCELoss(nn.Module):
    """
    Online hard example mining cross-entropy loss:在线难样本挖掘
    if loss[self.n_min] > self.thresh: 最少考虑 n_min 个损失最大的 pixel，
    如果前 n_min 个损失中最小的那个的损失仍然大于设定的阈值，
    那么取实际所有大于该阈值的元素计算损失:loss=loss[loss>thresh]。
    否则，计算前 n_min 个损失:loss = loss[:self.n_min]
    """
    def __init__(self, thresh, n_min, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()        # 将输入的概率转换为loss值
        self.n_min = n_min-1
        self.criteria = nn.CrossEntropyLoss(reduction='none').cuda()                    # 交叉熵损失
 
    def forward(self, logits, labels):
        loss = self.criteria(logits, labels)
        loss, _ = torch.sort(loss, descending=True)                                     # 由大到小排序
        if loss[self.n_min] > self.thresh:       
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        # print(loss.shape)
        return torch.mean(loss)
        
        
def SSVEP_train(targetS, datafolder, batch_size, lr, device, n_epoch, num_early_stop):
    tensorboardwriter = SummaryWriter('./logs')
    train_input, train_label, target_input, target_label = get_source_SSVEPdata(targetS=targetS, datafolder=datafolder)
  
    train_input = torch.from_numpy(train_input).float()                   
    train_label = torch.from_numpy(train_label).float()
    target_input = torch.from_numpy(target_input).float()
    target_label = torch.from_numpy(target_label).float()
    
    # 对训练集再划分出验证集
    X_train, X_val, y_train, y_val = train_test_split(train_input, train_label, test_size=0.2, random_state=0)
    print(X_train.shape)      
    print(y_train.shape)
        
    traindataset = torch.utils.data.TensorDataset(X_train, y_train)
    trainloader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=4)         
    valdataset = torch.utils.data.TensorDataset(X_val, y_val)
    valloader = torch.utils.data.DataLoader(dataset=valdataset, batch_size=batch_size, shuffle=True, num_workers=4)

    testdataset = torch.utils.data.TensorDataset(target_input, target_label)
    testloader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=1, shuffle=True, num_workers=4)

    model = Conformer().to(device)                        
    _, inch, h, w = train_input.shape
    summary(model, (batch_size, inch, h, w))

    loss_fn = torch.nn.CrossEntropyLoss().to(device)   
    celoss_fn_train = torch.nn.CrossEntropyLoss(reduction='none').to(device)

    val_max = 0
    early_stop_buffer = 0
    minest = 10000

    starttrain = time.time()
    for num_epoch in range(n_epoch):
        if (num_epoch % 10 == 0 and num_epoch > 0):
            lr = lr * 0.9
      
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-2)          
        model.train()             
        train_loss_sum, train_acc_sum, predict_loss_sum = 0, 0, 0
        one_epochsample = 0
        for i, (traindata, trainlabel) in enumerate(trainloader):     
            traindata = Variable(traindata.to(device))
            trainlabel = Variable(trainlabel.to(device))
            
            tok, trainouts = model(traindata)
            
            # convLSTM结合stablenet
            args = parser.parse_args()
            cfeatures = tok.mean(1)                   # model的两个输出：output（B，classes_num）和cfeatures（B，fc1.in_features）
            pre_features = model.pre_features.cuda()  # pre_features和pre_weight1在model文件定义
            pre_weight1 = model.pre_weight1.cuda()    

            if num_epoch >= args.epochp:              # epochp为预训练的epoch数
                # 对特征图进行RFF映射
                # pre_features 和pre_weight1 是预定义的
                weight1, pre_features, pre_weight1 = weight_learner(cfeatures, pre_features, pre_weight1, args, num_epoch, i)
            else:
                weight1 = Variable(torch.ones(cfeatures.size()[0], 1))  # .cuda())     # 预训练阶段权重为1

            model.pre_features.data.copy_(pre_features)
            model.pre_weight1.data.copy_(pre_weight1)

            #if num_epoch > 3:
            #    
            #    plt.hist(weight1.detach().cpu().numpy(),density=True)
            #    plt.xlabel('Weights')
            #    plt.ylabel('Probability')
            #    plt.title('StableNet Weights on Benchmark')
            #    plt.grid(True)
            #    plt.show()
            
            trainloss = celoss_fn_train(trainouts, trainlabel.long()).view(1, -1).mm(weight1.to(device)).view(1)
            #trainloss = loss_fn(trainouts, trainlabel.long())
            
            optimizer.zero_grad()
            one_epochsample += traindata.shape[0] 
 
            whole_loss = trainloss 
            whole_loss.backward()         
            optimizer.step()

            train_loss_sum += trainloss.cpu().item()
            train_pred = torch.max(trainouts, 1)[1]
            train_acc_sum += (train_pred == trainlabel).sum().cpu().item()

        
        train_loss = train_loss_sum / len(trainloader)
        # train_acc = train_acc_sum / len(traindataset) 
        train_acc = train_acc_sum / one_epochsample 

        # val
        val_loss_sum, val_acc_sum = 0, 0
        model.eval()
        with torch.no_grad():
            for i, (valdata, vallabel) in enumerate(valloader):
                valdata = Variable(valdata.to(device))
                vallabel = Variable(vallabel.to(device))
                    
                tok, valouts = model(valdata)
                valloss = loss_fn(valouts, vallabel.long())
                     
                ## 中间可视化tok
                ## print(tok.shape)     # [64, 800]
                #tok = tok.view(-1, valdata.shape[-1])
                #show_hook = tok.cpu().mean(dim=0)
                #plt.plot(show_hook)
                #plt.show()
    
                val_loss_sum += valloss.cpu().item()
                val_pred = torch.max(valouts, 1)[1]
                val_acc_sum += (val_pred == vallabel).sum().cpu().item()

        val_loss = val_loss_sum / len(valloader)
        val_acc = val_acc_sum / len(valdataset)

        # 以下是把验证集和训练集相同指标放在同一张图
        tensorboardwriter.add_scalars('acc', {'val_acc': val_acc}, num_epoch)
        tensorboardwriter.add_scalars('acc', {'train_acc':train_acc}, num_epoch)
        tensorboardwriter.add_scalars('loss', {'val_loss': val_loss}, num_epoch)
        tensorboardwriter.add_scalars('loss', {'train_loss':train_loss}, num_epoch)

        print('Epoch:', num_epoch,
              'Val loss: %.6f' % val_loss,
              'Val accuracy %.6f' % val_acc,
              'Train loss: %.6f' % train_loss,
              'Train accuracy %.6f' % train_acc)

        if (val_acc > val_max):
            val_max = val_acc
            #torch.save(model.state_dict(), 'forreviewer8_response/{0}_{1}.pth'.format(targetS, train_input.shape[-1]))
            torch.save(model.state_dict(), 'tmp.pth')

        if (val_loss < minest):
            minest = val_loss
            early_stop_buffer = 0
        early_stop_buffer = early_stop_buffer + 1
        if (early_stop_buffer == num_early_stop):      # 通过val_loss来判断早停
            break
            
    plt.figure(figsize=(8, 6))
    plt.hist(weight1.detach().cpu().numpy())
    plt.xlabel('Weights', fontsize=18)
    plt.ylabel('Probability',fontsize=18)
    # plt.title('StableNet Weights on Benchmark')
    plt.grid(True)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=16)
    #plt.savefig(r'benchmarkweight_S1.eps', bbox_inches='tight',dpi=500)
    plt.show()
    
    endtrain = time.time()
    # test
    test_correct = 0
    test_loss_sum = 0
    #model.load_state_dict(torch.load('forreviewer8_response/{0}_{1}.pth'.format(targetS, train_input.shape[-1]), map_location=device))
    model.load_state_dict(torch.load('tmp.pth', map_location=device))
    
    model.eval()
    starttest = time.time()
    with torch.no_grad():
        for i, (testdata, testlabel) in enumerate(testloader):    
            testdata = Variable(testdata.to(device))
            testlabel = Variable(testlabel.to(device))
            testtok, testout = model(testdata)
            
            ## 可视化测试集中间层输出
            #intermediate_visualize(testdata.detach().cpu(), testlabel.detach().cpu())
            #intermediate_visualize(testtok.detach().cpu(), testlabel.detach().cpu())
            #intermediate_visualize(testout.detach().cpu(), testlabel.detach().cpu())

            testpred = torch.max(testout, 1)[1]                       
            # print(testpred)
            test_correct += (testpred == testlabel).sum().cpu().item()

    endtest = time.time()
    testtime = endtest-starttest
    test_acc = test_correct / len(testdataset)
    print('Test accuracy is %.6f' % test_acc)
    return test_acc, testtime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    setup_seed(3407)
    
    # 完整被试列表
    S_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20', 'S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30', 'S31', 'S32', 'S33', 'S34', 'S35']
    
    acc_list = []
    traintime_list = []
    for n_S in range(len(S_list)):
        targetS = S_list[n_S]
        print('targetS:', targetS)
        datafolder = r"/home/ljw/dataset_opensrc/Tsinghua BCI Lab Benchmark Dataset/"
        singleS_acc, traintime = SSVEP_train(targetS=targetS, datafolder=datafolder, batch_size=64, lr=0.001, device="cuda", n_epoch=100, num_early_stop=10)
        acc_list.append(singleS_acc)
        traintime_list.append(traintime)
        print(time.asctime(time.localtime(time.time())))
        print(acc_list)
        print(np.mean(acc_list))
        print(traintime_list)
    print(acc_list)
    print('average train time', np.mean(traintime_list))