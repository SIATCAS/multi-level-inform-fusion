import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler 
import torch,re,os
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import  train_test_split
from sklearn.utils import shuffle
import scipy.io as scio
import torch.nn as nn
import gc, math 
torch.__version__

torch.manual_seed(128)
np.random.seed(1024)

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def RMSE_MARD(pred_values,ref_values):
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    data_length = len(ref_values)
    total = 0
    for i in range (data_length):
        temp = (ref_values[i] - pred_values[i]) * (ref_values[i] - pred_values[i])
        total = total + temp

    smse_value = math.sqrt(total / data_length)
    print('RMSE: ',smse_value)
    
    total = 0
    for i in range(data_length):
        temp = abs((ref_values[i] - pred_values[i]) / ref_values[i])
        total = total + temp
    mard_value = total / data_length
    print('MARD: ',mard_value)
        
    return smse_value,mard_value


def CONCOR(pred_values,ref_values):    
    data_length = len(ref_values)
    pred_values = pred_values.reshape(-1)
    ref_values = ref_values.reshape(-1)
    M1 = 0
    M2 = 0
    M3 = 0
    for i in range (data_length):
        v1 = abs(ref_values[i] - pred_values[i])
        v2 = v1 / ref_values[i]
        if v1 <= 15 or v2 <= 0.15:
           M1 = M1 + 1
        if v1 <= 20 or v2 <= 0.2:
            M2 = M2 + 1
        if v1 <= 30 or v2 <= 0.3:
            M3 = M3 + 1
    per_M1 = M1 / data_length
    per_M2 = M2 / data_length
    per_M3 = M3 / data_length
    print('per_M1:{:.4f}, per_M2:{:.4f}, per_M3:{:.4f}'.format(per_M1, per_M2, per_M3)) 
    per_M = np.matrix([per_M1, per_M2, per_M3]) 
    return per_M


def clarke_error_zone_detailed(act, pred):
    if (act < 70 and pred < 70) or abs(act - pred) < 0.2 * act:
        return 0
    # Zone E - left upper
    if act <= 70 and pred >= 180:
        return 8
    # Zone E - right lower
    if act >= 180 and pred <= 70:
        return 7
    # Zone D - right
    if act >= 240 and 70 <= pred <= 180:
        return 6
    # Zone D - left
    if act <= 70 <= pred <= 180:
        return 5
    # Zone C - upper
    if 70 <= act <= 290 and pred >= act + 110:
        return 4
    # Zone C - lower
    if 130 <= act <= 180 and pred <= (7/5) * act - 182:
        return 3
    # Zone B - upper
    if act < pred:
        return 2
    # Zone B - lower
    return 1

def parkes_error_zone_detailed(act, pred, diabetes_type):
    def above_line(x_1, y_1, x_2, y_2, strict=False):
        if x_1 == x_2:
            return False

        y_line = ((y_1 - y_2) * act + y_2 * x_1 - y_1 * x_2) / (x_1 - x_2)
        return pred > y_line if strict else pred >= y_line

    def below_line(x_1, y_1, x_2, y_2, strict=False):
        return not above_line(x_1, y_1, x_2, y_2, not strict)

    def parkes_type_1(act, pred):
        # Zone E
        if above_line(0, 150, 35, 155) and above_line(35, 155, 50, 550):
            return 7
        # Zone D - left upper
        if (pred > 100 and above_line(25, 100, 50, 125) and
                above_line(50, 125, 80, 215) and above_line(80, 215, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 550, 150)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 50, 80) and
                above_line(50, 80, 70, 110) and above_line(70, 110, 260, 550)):
            return 4
        # Zone C - right lower
        if (act > 120 and below_line(120, 30, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 140, 170) and
                above_line(140, 170, 280, 380) and (act < 280 or above_line(280, 380, 430, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 170, 145) and
                below_line(170, 145, 385, 300) and (act < 385 or below_line(385, 300, 550, 450))):
            return 1
        # Zone A
        return 0

    def parkes_type_2(act, pred):
        # Zone E
        if (pred > 200 and above_line(35, 200, 50, 550)):
            return 7
        # Zone D - left upper
        if (pred > 80 and above_line(25, 80, 35, 90) and above_line(35, 90, 125, 550)):
            return 6
        # Zone D - right lower
        if (act > 250 and below_line(250, 40, 410, 110) and below_line(410, 110, 550, 160)):
            return 5
        # Zone C - left upper
        if (pred > 60 and above_line(30, 60, 280, 550)):
            return 4
        # Zone C - right lower
        if (below_line(90, 0, 260, 130) and below_line(260, 130, 550, 250)):
            return 3
        # Zone B - left upper
        if (pred > 50 and above_line(30, 50, 230, 330) and
                (act < 230 or above_line(230, 330, 440, 550))):
            return 2
        # Zone B - right lower
        if (act > 50 and below_line(50, 30, 90, 80) and below_line(90, 80, 330, 230) and
                (act < 330 or below_line(330, 230, 550, 450))):
            return 1
        # Zone A
        return 0

    if diabetes_type == 1:
        return parkes_type_1(act, pred)

    if diabetes_type == 2:
        return parkes_type_2(act, pred)

    raise Exception('Unsupported diabetes type')

clarke_error_zone_detailed = np.vectorize(clarke_error_zone_detailed)
parkes_error_zone_detailed = np.vectorize(parkes_error_zone_detailed)

def zone_accuracy(act_arr, pred_arr, mode='clarke', detailed=False, diabetes_type=1):
    """
    Calculates the average percentage of each zone based on Clarke or Parkes
    Error Grid analysis for an array of predictions and an array of actual values
    """
    acc = np.zeros(9)
    if mode == 'clarke':
        res = clarke_error_zone_detailed(act_arr, pred_arr)
    elif mode == 'parkes':
        res = parkes_error_zone_detailed(act_arr, pred_arr, diabetes_type)
    else:
        raise Exception('Unsupported error grid mode')

    acc_bin = np.bincount(res)
    acc[:len(acc_bin)] = acc_bin

    if not detailed:
        acc[1] = acc[1] + acc[2]
        acc[2] = acc[3] + acc[4]
        acc[3] = acc[5] + acc[6]
        acc[4] = acc[7] + acc[8]
        acc = acc[:5]
    score = acc / sum(acc)
    print('CEG:  A:{:.4f}, B:{:.4f}, C:{:.4f}, D:{:.4f}, E:{:.4f}'.format(score[0], score[1], score[2], score[3],score[4]))
    return score

def sources_and_subsets_nodes(N):
    str1 = "{0:{fill}"+str(N)+"b}"
    a = []
    for i in range(1,2**N):
        a.append(str1.format(i, fill='0'))

    sourcesInNode = []
    sourcesNotInNode = []
    subset = []
    sourceList = list(range(N))
    def node_subset(node, sourcesInNodes):
        return [node - 2**(i) for i in sourcesInNodes]
    
    def string_to_integer_array(s, ch):
        N = len(s) 
        return [(N - i - 1) for i, ltr in enumerate(s) if ltr == ch]
    
    for j in range(len(a)):
        idxLR = string_to_integer_array(a[j],'1')
        sourcesInNode.append(idxLR)  
        sourcesNotInNode.append(list(set(sourceList) - set(idxLR)))
        subset.append(node_subset(j,idxLR))

    return sourcesInNode, subset


def subset_to_indices(indices):
    return [i for i in indices]

class Choquet_integral(torch.nn.Module):
    
    def __init__(self, N_in, N_out):
        super(Choquet_integral,self).__init__()
        self.N_in = N_in
        self.N_out = N_out
        self.nVars = 2**self.N_in - 2
        
        dummy = (1./self.N_in) * torch.ones((self.nVars, self.N_out), requires_grad=True)
        self.vars = torch.nn.Parameter(dummy)
        
        self.sourcesInNode, self.subset = sources_and_subsets_nodes(self.N_in)
        
        self.sourcesInNode = [torch.tensor(x) for x in self.sourcesInNode]
        self.subset = [torch.tensor(x) for x in self.subset]
        
    def forward(self,inputs):    
        self.FM = self.chi_nn_vars(self.vars)
        sortInputs, sortInd = torch.sort(inputs,1, True)
        M, N = inputs.size()
        sortInputs = torch.cat((sortInputs, torch.zeros(M,1)), 1)
        sortInputs = sortInputs[:,:-1] -  sortInputs[:,1:]
        
        out = torch.cumsum(torch.pow(2,sortInd),1) - torch.ones(1, dtype=torch.int64)
        
        data = torch.zeros((M,self.nVars+1))
        
        for i in range(M):
            data[i,out[i,:]] = sortInputs[i,:] 
        
        
        ChI = torch.matmul(data,self.FM)
            
        return ChI
    
    def chi_nn_vars(self, chi_vars):
        chi_vars = torch.abs(chi_vars)
        
        FM = chi_vars[None, 0,:]
        for i in range(1,self.nVars):
            indices = subset_to_indices(self.subset[i])
            if (len(indices) == 1):
                FM = torch.cat((FM,chi_vars[None,i,:]),0)
            else:
                maxVal,_ = torch.max(FM[indices,:],0)
                temp = torch.add(maxVal,chi_vars[i,:])
                FM = torch.cat((FM,temp[None,:]),0)
              
        FM = torch.cat([FM, torch.ones((1,self.N_out))],0)
        FM = torch.min(FM, torch.ones(1))  
        
        return FM
    

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        # print(out.shape)
        return self.sigmoid(out)
  
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        # print(x.shape)
        return self.sigmoid(x)
    
    
class ConvBlock(nn.Module):
    def __init__(self, in_planes,in_kernel, in_channel, f, filters, p1, p2, p3, p4):
        super(ConvBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=(1,2), padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(in_kernel)        
        
        self.shortcut_1 = nn.Conv2d(in_channel, F3, kernel_size=(1,1), stride=(1,2), padding=p4, bias=False)
        self.batch_1 = nn.BatchNorm2d(F3)
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        # print(X.shape)
        X_shortcut = self.shortcut_1(X)
        # print(X_shortcut.shape)        
        X_shortcut = self.batch_1(X_shortcut)
        X = self.stage(X)
        # print(X.shape) 
        X = self.ca(X) * X       
        X = self.sa(X) * X   
        # print(X.shape)        
        X = X + X_shortcut
        X = self.relu_1(X)
        return X  
           
    
class IndentityBlock(nn.Module):
    def __init__(self,in_planes,in_kernel, in_channel, f, filters, p1, p2, p3):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,kernel_size=(1,1),stride=1, padding=p1, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=p2, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,kernel_size=(1,1),stride=1, padding=p3, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention(in_kernel)         
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        # print(X.shape) 
        X = self.stage(X)
        # print(X.shape) 
        X = self.ca(X) * X       
        X = self.sa(X) * X   
        # print(X.shape)         
        X = X + X_shortcut
        X = self.relu_1(X)
        return X
          
    
class ResModel(nn.Module):
    def __init__(self):
        super(ResModel,self).__init__()
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels =1,out_channels =8,kernel_size=(1,6),stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.MaxPool2d((1,2),padding=0),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(in_planes=16, in_kernel =5, in_channel=8, f=(1,6), filters=[8, 8, 16],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(in_planes=16, in_kernel=5, in_channel=16, f=(1,5), filters=[8, 8, 16],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=16, in_kernel=5, in_channel=16, f=(1,5), filters=[8, 8, 16],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(in_planes=32, in_kernel =5, in_channel=16, f=(1,6), filters=[8, 8, 32],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(in_planes=32, in_kernel=5,in_channel=32, f=(1,5), filters=[8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=32, in_kernel=5,in_channel=32, f=(1,5), filters=[8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=32, in_kernel=5,in_channel=32, f=(1,5), filters=[8, 8, 32],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(in_planes=64, in_kernel =5, in_channel=32, f=(1,6), filters=[16, 16, 64],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(in_planes=64, in_kernel=5, in_channel=64, f=(1,5), filters=[16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=64, in_kernel=5, in_channel=64, f=(1,5), filters=[16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=64, in_kernel=5, in_channel=64, f=(1,5), filters=[16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=64, in_kernel=5, in_channel=64, f=(1,5), filters=[16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=64, in_kernel=5, in_channel=64, f=(1,5), filters=[16, 16, 64],p1=(0,1),p2=(0,1),p3=0),
        )
        self.stage5 = nn.Sequential(
            ConvBlock(in_planes=128, in_kernel =5,in_channel=64, f=(1,6), filters=[32, 32, 128],p1=(0,0),p2=(0,1),p3=(0,2),p4=(0,1)),
            IndentityBlock(in_planes=128, in_kernel=5,in_channel=128, f=(1,5), filters=[32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
            IndentityBlock(in_planes=128, in_kernel=5,in_channel=128, f=(1,5), filters=[32, 32, 128],p1=(0,1),p2=(0,1),p3=0),
        )                                
        
        self.p1 = nn.Sequential(
             nn.Conv2d(32,128,kernel_size=(1,1),stride=1, padding=(0,4), bias=False),
             nn.MaxPool2d((1,4),padding=0)
             )
        self.p2 = nn.Sequential(
             nn.Conv2d(64,128,kernel_size=(1,1),stride=1, padding=(0,2), bias=False),
             nn.MaxPool2d((1,2),padding=0)
             )                 

        self.fc5_1 = nn.Linear(49152,3000)
        self.fc5_2 = nn.Dropout(0.3)
        self.fc5_3 = nn.Linear(3000,300)  
        self.fc5_4 = nn.Dropout(0.2)
        self.fc5_5 = nn.Linear(300,30)         
        self.fc5_6 = nn.Linear(30,1)            
    
    def forward(self, X):
        out1 = self.stage1(X)        
        out1 = self.stage2(out1)         
        out1 = self.stage3(out1)
        out1_s3 = out1     
        out1_s3 = self.p1(out1_s3) 
        out1_s3 = out1_s3.view(out1.size(0),128,-1)
        
        out1 = self.stage4(out1) 
        out1_s4 = out1
        out1_s4 = self.p2(out1_s4)
        out1_s4 = out1_s4.view(out1.size(0),128,-1)
              
        out1_s5 = self.stage5(out1)
        out1_s5 = out1_s5.view(out1.size(0),128,-1) 
        
        out1_34 = torch.bmm(out1_s4, torch.transpose(out1_s3, 1, 2))
        out1_34 = out1_34.view(out1.size(0), 128**2)
  #      out1_34 = torch.sqrt(out1_34 + 1e-5)
        out1_34 = torch.nn.functional.normalize(out1_34)         
               
        out1_35 = torch.bmm(out1_s5, torch.transpose(out1_s3, 1, 2))
        out1_35 = out1_35.view(out1.size(0), 128**2)
  #      out1_35 = torch.sqrt(out1_35 + 1e-5)
        out1_35 = torch.nn.functional.normalize(out1_35)        
                
        out1_45 = torch.bmm(out1_s5, torch.transpose(out1_s4, 1, 2))
        out1_45 = out1_45.view(out1.size(0), 128**2)
  #      out1_45 = torch.sqrt(out1_45 + 1e-5)
        out1_45 = torch.nn.functional.normalize(out1_45)         
                  
        out = torch.cat((out1_45,out1_35,out1_34),dim = 1)
        out = self.fc5_1(out)         
        out = self.fc5_2(out) 
        out = self.fc5_3(out) 
        out = self.fc5_4(out) 
        out = self.fc5_5(out) 
        out = self.fc5_6(out) 
        # print(out.shape)
        return  out
    
# resmodel = ResModel()
# resmodel(torch.rand(10,1,20,760)))

def tongji(data):
    P_len = 600
    num = np.unique(data)
    out_data = np.zeros((len(num),3))
    for t in range(len(num)):
        total = 0
        for m in range(len(data)):
            if data[m] == num[t]:
                total = total + 1
        out_data[t,0] = num[t] 
        out_data[t,1] = total
        out_data[t,2] = int(P_len / total)
    return out_data





numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

num = np.linspace(0,19,20)
train_data_ECG = np.zeros((1,760))
test_data_ECG = np.zeros((1,760))
train_label_list = []
test_label_list = []

path1 = '/data0/ljz/ECG_PPG/ECG_norm'


data_dir_list1 = sorted(os.listdir(path1),key=numericalSort) 
data_dir_s1 = shuffle(data_dir_list1, random_state = 8)


"""
第1折:  N > 9
第2折:  N < 10 or N > 19
第3折:  N < 20 or N > 29
第4折:  N < 30 or N > 39
第5折:  N < 40 or N > 49
第6折:  N < 50 or N > 59
第7折:  N < 60 or N > 69
第8折:  N < 70 or N > 79
第9折:  N < 80 or N > 89
第10折:  N < 90 or N > 99
"""

N = 0
for path in data_dir_s1:
    if N < 50 or N > 59:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["save_ECG"]
      temp_data = temp1[:,0:760]
      temp_label = temp1[:,760]
      train_data_ECG = np.vstack((train_data_ECG,temp_data)) 
      K = temp_data.shape[0]
      for k in range(K):
          if k % 20 == 0:
              train_label_list.append(temp_label[k])
    else:
      all_data = scio.loadmat(path1 + '/' + path)   
      temp1 = all_data["save_ECG"]
      temp_data = temp1[:,0:760]
      temp_label = temp1[:,760]
      test_data_ECG = np.vstack((test_data_ECG,temp_data))
      K = temp_data.shape[0]
      for k in range(K):   
          if k % 20 == 0:
              test_label_list.append(temp_label[k])        
    N = N + 1 

train_data_ECG = np.delete(train_data_ECG,[0],axis=0) 
train_data_ECG = train_data_ECG.reshape(-1,20,760)

test_data_ECG = np.delete(test_data_ECG,[0],axis=0)
test_data_ECG = test_data_ECG.reshape(-1,20,760)

train_label = np.array(train_label_list)
test_label = np.array(test_label_list)


train_add_ECG_list = []
train_add_label_list = []

pp = tongji(train_label)
for t in range(len(pp)):
    if pp[t,2] > 0:
       p_label = pp[t,0] 
       for m in range(len(train_label)):
           if p_label == train_label[m]:
               temp_ECG = train_data_ECG[m]
               for w in range(2,2+int(pp[t,2])):
                   num1 = shuffle(num, random_state=w)           
                   for u in range(len(num1)):
                       y = int(num1[u])
                       train_add_ECG_list.append(temp_ECG[y,:])
                   train_add_label_list.append(train_label[m])
                   
train_add_ECG = np.array(train_add_ECG_list)
train_add_ECG = train_add_ECG.reshape(-1,20,760)
train_new_ECG = np.vstack((train_data_ECG,train_add_ECG))


del train_add_ECG_list, train_add_ECG
gc.collect()

train_add_label =np.array(train_add_label_list)
train_add_label = train_add_label.reshape(-1,1)
train_label = train_label.reshape(-1,1)
train_new_label = np.vstack((train_label,train_add_label))
train_new_label = train_new_label.reshape(-1)

train_ID = np.arange(0,len(train_new_label),1)
train_use = np.arange(0,len(train_label),1)
test_ID = np.arange(0,len(test_label),1)

train_ID = torch.from_numpy(train_ID)
train_use = torch.from_numpy(train_use)
test_ID = torch.from_numpy(test_ID)

train_Label = torch.from_numpy(train_new_label)
use_Label = torch.from_numpy(train_label)
test_Label = torch.from_numpy(test_label)


BATCH_SIZE = 128
train_set = torch.utils.data.TensorDataset(train_ID, train_Label)
train_set2 = torch.utils.data.TensorDataset(train_use, use_Label)
test_set = torch.utils.data.TensorDataset(test_ID, test_Label)


train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(dataset=train_set2, batch_size=BATCH_SIZE, shuffle=False) 
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False) 

del train_set, train_set2, test_set
gc.collect()

resmodel = ResModel()


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
resmodel = resmodel.to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(resmodel.parameters(), lr=LEARNING_RATE)
TOTAL_EPOCHS = 40
losses = []

EPOCH_list = []
train_RMSE_list = []
train_MARD_list = []
test_RMSE_list = []
test_MARD_list = []
test_concor15_list = []
test_concor20_list = []
test_concor30_list = [] 
test_ClarkeA_list = [] 
test_ClarkeAB_list = [] 
test_CERA_list = [] 
test_CERAB_list = [] 



for epoch in range(TOTAL_EPOCHS):
    print('Epoch: %d' %(epoch + 1))
    train_pred_list = []
    train_true_list = []
    test_pred_list = []
    test_true_list = []
       
    for m, (train_images, tr_labels) in enumerate(train_loader):
        tr_labels = tr_labels.to(DEVICE).float()
        train_images = train_images.numpy()
        data_len = len(train_images)
        train_ECG_list = []
        for n in range(data_len):
            path_temp = train_images[n]
            temp_ECG = train_new_ECG[path_temp]
            train_ECG_list.append(temp_ECG)
        
        train_ECG = np.array(train_ECG_list)
        train_ECG = np.expand_dims(train_ECG, axis=1) 
        train_ECG = torch.from_numpy(train_ECG)
        train_ECG = train_ECG.float().to(DEVICE)        
      
        optimizer.zero_grad()
        train_outputs = resmodel(train_ECG)
        train_outputs = train_outputs.view(-1)
        train_loss = criterion(train_outputs,tr_labels)
        train_loss.backward()
        optimizer.step()
        losses.append(train_loss.cpu().data.item())
                
    
    resmodel.eval() 
    
    for m, (train_images, tr_labels) in enumerate(train_loader2):
        tr_labels = tr_labels.to(DEVICE).float()
        train_images = train_images.numpy()
        data_len = len(train_images)
        train_ECG_list = []
        for n in range(data_len):
            path_temp = train_images[n]
            temp_ECG = train_data_ECG[path_temp]
            train_ECG_list.append(temp_ECG)
        
        train_ECG = np.array(train_ECG_list)
        train_ECG = np.expand_dims(train_ECG, axis=1) 
        train_ECG = torch.from_numpy(train_ECG)
        train_ECG = train_ECG.float().to(DEVICE)        

        train_outputs = resmodel(train_ECG)
        train_outputs = train_outputs.view(-1) 
        Train_outputs = train_outputs.cpu()
        Train_outputs = Train_outputs.detach().numpy()
        Tr_labels = tr_labels.cpu()
        Tr_labels = Tr_labels.numpy() 
        
        for t in range(data_len):
            train_pred_list.append(Train_outputs[t])
            train_true_list.append(Tr_labels[t])
    
    train_pred = np.array(train_pred_list)
    train_true = np.array(train_true_list)
    train_result = RMSE_MARD(train_pred*18, train_true*18)       
    
    
    for n, (test_images, te_labels) in enumerate(test_loader):
        te_labels = te_labels.float()
        test_images = test_images.numpy()
        data_len = len(test_images)
        test_ECG_list = []
        for n in range(data_len):
            path_temp = test_images[n]
            temp_ECG = test_data_ECG[path_temp]
            test_ECG_list.append(temp_ECG)
        
        test_ECG = np.array(test_ECG_list)
        test_ECG = np.expand_dims(test_ECG, axis=1) 
        test_ECG = torch.from_numpy(test_ECG)
        test_ECG = test_ECG.float().to(DEVICE)        

        test_outputs = resmodel(test_ECG)
        test_outputs = test_outputs.view(-1)                
        test_outputs = test_outputs.cpu()
        test_outputs = test_outputs.detach().numpy()
        Te_labels = te_labels.numpy() 
        
        for t in range(data_len):
            test_pred_list.append(test_outputs[t])
            test_true_list.append(Te_labels[t])
    
    test_pred = np.array(test_pred_list)
    test_true = np.array(test_true_list)
    test_result = RMSE_MARD(test_pred*18, test_true*18) 
    print("---------------------------")
    
    N = 7
    num_train = len(train_true) // N
    total_train = int(num_train * N)
    Train_Pred1_1_list = []
    Train_Label_list = []

    for m in range(total_train - N):
        temp = train_pred[m:m+N].reshape(1,-1)
        Train_Pred1_1_list.append(temp)
        Train_Label_list.append(train_true[m+N-1])
    
    Train_Pred1_1 = np.array(Train_Pred1_1_list)
    Train_Pred1_1 = Train_Pred1_1.reshape(-1,N)
    Train_Label = np.array(Train_Label_list) 
   
    
    num_test = len(test_true) // N
    total_test = int(num_test * N)
    Test_Pred1_1_list = [] 
    Test_Label_list = []
    for m in range(total_test - N):
        temp = test_pred[m:m+N].reshape(1,-1)
        Test_Pred1_1_list.append(temp) 
        Test_Label_list.append(test_true[m+N-1])
    
    Test_Pred1_1 = np.array(Test_Pred1_1_list)
    Test_Pred1_1 = Test_Pred1_1.reshape(-1,N)
    Test_Label = np.array(Test_Label_list)
    
    Train_Pred1_1 = torch.tensor(Train_Pred1_1,dtype=torch.float)
    Train_value = torch.tensor(Train_Label,dtype=torch.float)
    Test_Pred1_1 = torch.tensor(Test_Pred1_1,dtype=torch.float)
    Test_value = torch.tensor(Test_Label,dtype=torch.float)
    
    net1_1 = Choquet_integral(N_in=N,N_out=1)
    learning_rate = 0.001
    criterion1_1 = torch.nn.MSELoss(reduction='mean')
    optimizer1_1 = torch.optim.Adam(net1_1.parameters(), lr=learning_rate) 
    num_epochs = 2

    for epo in range(num_epochs):
        print('epo: %d' %(epo + 1))    
        optimizer1_1.zero_grad()
        Train_P1_1 = net1_1(Train_Pred1_1)
        Train_P1_1 = Train_P1_1.view(-1)
        Train_value = Train_value.view(-1)
        loss1_1 = criterion1_1(Train_P1_1, Train_value)
        loss1_1.backward()
        optimizer1_1.step()     
        Train_P1_1 = Train_P1_1.cpu().detach().numpy()   
        train_result1_1 = RMSE_MARD(Train_P1_1*18, Train_Label*18) 
        
        net1_1.eval()
        test_1_1 = net1_1(Test_Pred1_1)
        test_1_1 = test_1_1.cpu().detach().numpy()
        test_1_1 = test_1_1.reshape(-1)   
        Test_Label = Test_Label.reshape(-1)
        test_result1_1 = RMSE_MARD(test_1_1*18, Test_Label*18) 
        test_concor = CONCOR(test_1_1*18, Test_Label*18)
        test_Clarke = zone_accuracy(Test_Label*18, test_1_1*18, mode='clarke', detailed=False)
        test_CER = zone_accuracy(Test_Label*18, test_1_1*18, mode='parkes',diabetes_type=1, detailed=False)        
    
    
    EPOCH_list.append(epoch + 1)  
    train_RMSE_list.append(train_result1_1[0])  
    train_MARD_list.append(train_result1_1[1])  
    test_RMSE_list.append(test_result1_1[0])  
    test_MARD_list.append(test_result1_1[1])  
    test_concor15_list.append(test_concor[0,0])
    test_concor20_list.append(test_concor[0,1])
    test_concor30_list.append(test_concor[0,2])    
    test_ClarkeA_list.append(test_Clarke[0])
    test_ClarkeAB_list.append(test_Clarke[0] + test_Clarke[1])
    test_CERA_list.append(test_CER[0])
    test_CERAB_list.append(test_CER[0] + test_CER[1])       
        
    
    EPOCH = np.array(EPOCH_list)    
    train_RMSE = np.array(train_RMSE_list)
    train_MARD = np.array(train_MARD_list)
    test_RMSE = np.array(test_RMSE_list)
    test_MARD = np.array(test_MARD_list)  
    test_concor15 = np.array(test_concor15_list)
    test_concor20 = np.array(test_concor20_list)
    test_concor30 = np.array(test_concor30_list)
    test_ClarkeA = np.array(test_ClarkeA_list)
    test_ClarkeAB = np.array(test_ClarkeAB_list)
    test_CERA = np.array(test_CERA_list)
    test_CERAB = np.array(test_CERAB_list)  

    EPOCH = EPOCH.reshape(-1,1)
    train_RMSE = train_RMSE.reshape(-1,1)
    train_MARD = train_MARD.reshape(-1,1) 
    test_RMSE = test_RMSE.reshape(-1,1)
    test_MARD = test_MARD.reshape(-1,1)      
    test_concor15 = test_concor15.reshape(-1,1)
    test_concor20 = test_concor20.reshape(-1,1) 
    test_concor30 = test_concor30.reshape(-1,1)
    test_ClarkeA = test_ClarkeA.reshape(-1,1)
    test_ClarkeAB = test_ClarkeAB.reshape(-1,1) 
    test_CERA = test_CERA.reshape(-1,1)
    test_CERAB = test_CERAB.reshape(-1,1)      
    
    save_result = np.hstack((EPOCH, train_RMSE, train_MARD, test_RMSE, test_MARD,test_concor15, test_concor20, test_concor30, 
                             test_ClarkeA, test_ClarkeAB, test_CERA, test_CERAB))    
    np.savetxt('/data0/ljz/ECG_PPG/BG result/ECG_Fold 6.csv', save_result, delimiter=',', fmt='%1.5f')
          