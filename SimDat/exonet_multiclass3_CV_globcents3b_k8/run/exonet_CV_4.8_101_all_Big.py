### python packages
import os
from os import path
import numpy as np
import glob as glob
from random import random
import pandas as pd
import pickle
import time

### torch packages
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler

### sklearn packages
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold


### remove these later (for notebook version only)
'''
from tqdm import tqdm_notebook as tqdm
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, export_png
from bokeh.layouts import row
output_notebook()
import matplotlib.pyplot as plt
import seaborn as sns
'''

torch.cuda.manual_seed(42)

class KeplerDataLoaderCrossVal(Dataset):
    
    '''
    
    PURPOSE: DATA LOADER FOR KERPLER LIGHT CURVES
    INPUT: PATH TO DIRECTOR WITH LIGHT CURVES + INFO FILES
    OUTPUT: LOCAL + GLOBAL VIEWS, LABELS
    
    '''

    def __init__(self, infofiles):
        ### list of global, local, and info files (assumes certain names of files)
        self.flist_global,self.flist_local=[],[]
        for i,val in enumerate(infofiles):
            self.flist_global.append(val.replace('_info2.npy','_glob.npy'))
            self.flist_local.append(val.replace('_info2.npy','_loc.npy'))
        self.flist_info = infofiles
        
        ### list of whitened centroid files
        #self.flist_global_cen = np.sort(glob.glob(os.path.join(filepath, '*global_cen_w.npy')))
        #self.flist_local_cen = np.sort(glob.glob(os.path.join(filepath, '*local_cen_w.npy')))
        
        ### ids = {TIC}_{TCE}
        self.ids = np.sort(['_'.join(x.split('/')[-1].split('_')[1:3]) for x in self.flist_global])

    def __len__(self):

        return self.ids.shape[0]

    def __getitem__(self, idx):

        ### grab local and global views
        data_global = np.nan_to_num(np.load(self.flist_global[idx],encoding='latin1'))
        data_local = np.nan_to_num(np.load(self.flist_local[idx],encoding='latin1'))

        ### grab centroid views
        data_global_cen = data_global[:,1]
        data_local_cen = data_local[:,1]
        
        data_global = data_global[:,0]
        data_local = data_local[:,0]
        
        ### info file contains: [0]kic, [1]tce, [2]period, [3]epoch, [4]duration, [5]label)
        data_info = np.load(self.flist_info[idx],encoding='latin1')
        #np.load(self.flist_info[idx],encoding='latin1')
        
        if data_info[6]=='PL':
            label=1
        elif data_info[6]=='UNK':
            label=0
        else:
            label=2
        
        #collist=['TPERIOD','TDUR','DRRATIO','NTRANS','TSNR','TDEPTH','INDUR',
        #         'SESMES_LOG_RATIO','PRAD_LOG_RATIO','TDUR_LOG_RATIO','RADRATIO','IMPACT',
        #         'TESSMAG','RADIUS','PMTOTAL','LOGG','MH','TEFF']#from bls search, derived from transit model, from starpars
        stelpars=np.nan_to_num(np.hstack((data_info[7:13].astype(float),data_info[-18:-6].astype(float))))
        
        return (data_local.astype(float), data_global.astype(float), data_local_cen.astype(float), data_global_cen.astype(float), stelpars), label
    
class BalancedBatchSampler(Sampler):
    """Wraps another sampler to yield a class-balanced mini-batch of indices.

    Args:
        batch_size (int): Size of mini-batch.
        label_index (int): The index of the label in the dataset returned tuple
        shuffle (bool): Whether to shuffle every iteration or not.
    """
    def __init__(self, batch_size, dataset, n_classes, label_index=1, shuffle=True, stats_file=None):
        self.n_classes = n_classes
        self.dataset = dataset
        self.label_index = label_index
        self.batch_size = batch_size
        self.class_indices = []
        self.class_iteration_index = [0]*self.n_classes
        self.max_class_length = 0
        self.max_class = 0
        self.per_class_batchsize = self.batch_size // self.n_classes
        self.shuffle = shuffle

        for i in range(self.n_classes):
            self.class_indices.append([])

        if stats_file is not None and os.path.exists(stats_file):
            with open(stats_file, 'rb') as stats_f:
                self.class_indices = pickle.load(stats_f)
            print('Loaded dataset class distribution from: {}'.format(stats_file))
        else:
            print('Accumulating dataset class distribution...')
            for i in np.arange(len(self.dataset)):
                label = int(self.dataset[i][self.label_index])
                self.class_indices[label].append(i)
            if stats_file is not None:
                with open(stats_file, 'wb') as stats_f:
                    pickle.dump(self.class_indices, stats_f)
                print('Saved dataset class distribution to: {}'.format(stats_file))

        for i in range(self.n_classes):
            if self.shuffle:
                self.class_indices[i] = list(np.random.permutation(self.class_indices[i]))

            if len(self.class_indices) > self.max_class_length:
                self.max_class_length = len(self.class_indices[i])
                self.max_class = i
        
        print('Balancing dataset with class distribution: {}'.format([len(class_ind) for class_ind in self.class_indices]))

    def __iter__(self):
        # only generate full batches
        while self.class_iteration_index[self.max_class] + self.per_class_batchsize  < self.max_class_length:
            batch = []

            random_class_indices = []
            for i in range(self.n_classes):
                # if not majority class, and we ran out of samples, reset
                if i!= self.max_class and len(self.class_indices[i]) < self.class_iteration_index[i] + self.per_class_batchsize:
                    self.class_iteration_index[i] = 0
                j = self.class_iteration_index[i]
                sliced = self.class_indices[i][j:j+self.per_class_batchsize] 
                batch.extend(sliced)
                self.class_iteration_index[i] += self.per_class_batchsize

            yield batch
            batch = []

        # reset indices for next epoch
        for i in range(self.n_classes):
            self.class_iteration_index[i] = 0
            if self.shuffle:
                self.class_indices[i] = list(np.random.permutation(self.class_indices[i]))

    def __len__(self):
        return self.max_class_length // self.per_class_batchsize
    
class Model(nn.Module):

    '''
    
    PURPOSE: DEFINE EXTRANET MODEL ARCHITECTURE
    INPUT: GLOBAL + LOCAL LIGHT CURVES AND CENTROID CURVES, STELLAR PARAMETERS
    OUTPUT: BINARY CLASSIFIER
    
    '''
    
    def __init__(self):

        ### initialize model
        super(Model, self).__init__()

        ### define global convolutional lalyer
        self.fc_global = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(64, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
            nn.Conv1d(128, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(256, 256, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=2),
        )

        ### define local convolutional lalyer
        self.fc_local = nn.Sequential(
            nn.Conv1d(2, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
            nn.Conv1d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(7, stride=2),
        )

        ### define fully connected layer that combines both views
        self.final_layer = nn.Sequential(
            nn.Linear(7858, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            ### need output of 1 because using BCE for loss
            nn.Linear(256, 3),
            nn.Softmax(1))

    def forward(self, x_local, x_local_cen, x_global, x_global_cen, x_star):
        
        x_local_all = torch.cat([x_local, x_local_cen], dim=1)
        x_global_all = torch.cat([x_global, x_global_cen], dim=1)

        ### get outputs of global and local convolutional layers
        out_global = self.fc_global(x_global_all)
        out_local = self.fc_local(x_local_all)
        
        ### flattening outputs (multi-dim tensor) from convolutional layers into vector
        out_global = out_global.view(out_global.shape[0], -1)
        out_local = out_local.view(out_local.shape[0], -1)

        ### join two outputs together
        out = torch.cat([out_global, out_local, x_star.squeeze(1)], dim=1)
        out = self.final_layer(out)

        return out
    
def invert_tensor(tensor):
    
    '''
    
    PURPOSE: FLIP A 1D TENSOR ALONG ITS AXIS
    INPUT: 1D TENSOR
    OUTPUT: INVERTED 1D TENSOR
    
    '''
    
    idx = [i for i in range(tensor.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx).cuda()
    inverted_tensor = tensor.index_select(0, idx)
    
    return inverted_tensor

def roll_tensor(tensor, shift):
    
    return torch.cat([tensor[shift:tensor.size(0)],tensor[:shift]])

def roll_array(array, shift):
    
    return np.hstack((array[shift:len(array)],array[:shift]))

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    if type(labels)==list or type(labels)==np.ndarray:
        target=np.zeros((len(labels),C))
        target[np.arange(len(labels)), np.array(labels).astype(int)] = 1
    else:
        one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        target = one_hot.scatter_(1, labels.data, 1)

        target = Variable(target)
        
    return target

def train_model(n_epochs, kepler_data_loader, kepler_val_loader, model, criterion, optimizer, augment='',savefile=None, cont=None):

    '''
    PURPOSE: TRAIN MODEL 
    
    INPUTS:  num_epoch = number of epochs for training
             kepler_data_loader = data loader for Kepler dataset
             model = model use for training
             criterion = criterion for calculating loss
             
    OUTPUT:  epoch_{train/val}_loss = training set loss for each epoch
             epoch_val_acc = validation set accuracy for each epoch
             epoch_val_ap = validation set avg. precision for each epoch
             final_val_pred = validation predictions from final model
             final_val_gt = validation ground truths
    '''
    
    if cont is None:
        minloss=1e9;bests=[];startepoch=0

        ### empty arrays to fill per-epoch outputs
        epoch_train_loss = [];      epoch_val_loss = []       ;epoch_val_ap=[]

        epoch_val_acc_pl  = []   ;  epoch_val_acc_ebs  = []   ; epoch_val_acc_unk = []    ; #epoch_val_acc_bebs = []
        epoch_val_recall_pl  = [];  epoch_val_recall_ebs= []  ; epoch_val_recall_unk = [] ; #epoch_val_recall_bebs = []
        epoch_val_ap_pl = []     ;  epoch_val_ap_ebs = []     ; epoch_val_ap_unk = []     ; #epoch_val_ap_bebs = []
    else:
        minloss=np.min(cont['loss_val_epoch'])
        bests=[cont['pred_val_final'],cont['gt_val_final']]
        print('cont:',cont.keys(),('epoch' in cont.keys()))
        if 'epoch' not in cont.keys():
            cont['epoch']=len(cont['loss_val_epoch'])
        startepoch=cont['epoch']
        
        ### empty arrays to fill per-epoch outputs
        epoch_train_loss = cont['loss_train_epoch']        ;  epoch_val_loss = cont['loss_val_epoch']
        epoch_val_ap=cont['ap_val_epoch']

        epoch_val_acc_pl  = cont['acc_val_epoch_pl']       ;  epoch_val_acc_ebs  = cont['acc_val_epoch_ebs']
        epoch_val_acc_unk = cont['acc_val_epoch_unk']      ;  #epoch_val_acc_bebs = cont['acc_val_epoch_bebs']
        epoch_val_recall_pl  = cont['recall_val_epoch_pl'] ;  epoch_val_recall_ebs= cont['recall_val_epoch_ebs'] 
        epoch_val_recall_unk = cont['recall_val_epoch_unk'];  #epoch_val_recall_bebs = cont['recall_val_epoch_bebs']
        
        epoch_val_ap_pl = cont['ap_val_epoch_pl']
        epoch_val_ap_ebs = cont['ap_val_epoch_ebs'] if 'ap_val_epoch_ebs' in cont else []
        epoch_val_ap_unk = cont['ap_val_epoch_unk'] if 'ap_val_epoch_unk' in cont else []
        #epoch_val_ap_bebs = cont['ap_val_epoch_bebs'] if 'ap_val_epoch_bebs' in cont else []
    
    t0=time.time()
    ### loop over number of epochs of training
    for epoch in range(startepoch,startepoch+n_epochs):

        ####################
        ### for training set
        
        ### loop over batches
        train_loss = torch.zeros(1).cuda()
        n_train=0
        for x_train_data, y_train in kepler_data_loader:
            
            ### get local view, global view, and label for training
            x_train_local, x_train_global, x_train_local_cent,x_train_global_cent, x_train_star = x_train_data
                        
            x_train_local = Variable(x_train_local).type(torch.FloatTensor).cuda()
            x_train_local_cent = Variable(x_train_local_cent).type(torch.FloatTensor).cuda()
            x_train_global = Variable(x_train_global).type(torch.FloatTensor).cuda()    
            x_train_global_cent = Variable(x_train_global_cent).type(torch.FloatTensor).cuda()
            x_train_star = Variable(x_train_star).type(torch.FloatTensor).cuda()        
            y_train = Variable(y_train).type(torch.LongTensor).cuda()            
            
            if 'all' in augment and 'noise' not in augment:
                local_stds=x_train_local.cpu().numpy()
                local_roundn=int(np.floor(len(local_stds[0])*0.35))
                local_stds=np.hstack((local_stds[:,:local_roundn],local_stds[:,len(local_stds[0])-local_roundn:]))
                local_stds=np.nanstd(local_stds,axis=0)
                
                local_cent_stds=x_train_local_cent.cpu().numpy()
                #print('cents:',np.shape(local_cent_stds), local_cent_stds[:,0],'local:',np.shape(x_train_local.cpu().numpy()),x_train_local.cpu().numpy()[:,0])
                local_cent_stds=np.hstack((local_cent_stds[:,:local_roundn],local_cent_stds[:,len(local_cent_stds[0])-local_roundn:]))
                local_cent_stds=np.nanstd(local_cent_stds,axis=0)

                global_stds=x_train_global.cpu().numpy()
                global_roundn=int(np.floor(len(global_stds[0])*0.35))
                global_stds=np.hstack((global_stds[:,:global_roundn],global_stds[:,len(global_stds[0])-global_roundn:]))
                global_stds=np.nanstd(global_stds,axis=0)
                
                global_cent_stds=x_train_global_cent.cpu().numpy()
                #print('cents:',np.shape(local_cent_stds), local_cent_stds[:,0],'local:',np.shape(x_train_local.cpu().numpy()),x_train_local.cpu().numpy()[:,0])
                global_cent_stds=np.hstack((global_cent_stds[:,:global_roundn],global_cent_stds[:,len(global_cent_stds[0])-global_roundn:]))
                global_cent_stds=np.nanstd(global_cent_stds,axis=0)

            ### randomly invert half of light curves
            for batch_ind in range(x_train_local.shape[0]):
                if np.sum(np.isnan(x_train_local[batch_ind].cpu().numpy()))>0 or np.sum(np.isnan(x_train_global[batch_ind].cpu().numpy()))>0:
                    print(n_train,x_train_local[batch_ind].cpu().numpy())
                    raise ValueError("Nans present in arrays")
                
                if 'all' in augment and 'noise' not in augment:                    
                    sig_noise = abs(np.random.normal(0, 0.66))
                    local_noise = Variable(x_train_local[batch_ind].data.new(x_train_local[batch_ind].size()).normal_(0.0, sig_noise*local_stds[batch_ind]))
                    x_train_local[batch_ind] = x_train_local[batch_ind] + local_noise
                    global_noise = Variable(x_train_global[batch_ind].data.new(x_train_global[batch_ind].size()).normal_(0.0, sig_noise*global_stds[batch_ind]))
                    
                    cent_noise=Variable(x_train_local_cent[batch_ind].data.new(x_train_local_cent[batch_ind].size()).normal_(0.0, sig_noise*local_cent_stds[batch_ind]))
                    x_train_local_cent[batch_ind] = x_train_local_cent[batch_ind] + cent_noise
                    glob_cent_noise = Variable(x_train_global_cent[batch_ind].data.new(x_train_global_cent[batch_ind].size()).normal_(0.0,  sig_noise*global_cent_stds[batch_ind]))

                    x_train_global_cent[batch_ind] = x_train_global_cent[batch_ind] + glob_cent_noise

                '''
                if 'all' in augment or 'yshift' in augment:
                    ### add random gaussian scaling
                    scale_factor = torch.FloatTensor(1).normal_(0.0, 0.125).cuda()
                    x_train_local[batch_ind] = x_train_local[batch_ind] + scale_factor
                    x_train_global[batch_ind] = x_train_global[batch_ind] + scale_factor
                '''
                if 'all' in augment and 'xshift' not in augment:
                   ### shift by some random iteger
                    shift_local = np.random.randint(-5, high=5)
                    x_train_local[batch_ind] = roll_tensor(x_train_local[batch_ind], shift_local)
                    shift_global = np.random.randint(-30, high=30)
                    x_train_global[batch_ind] = roll_tensor(x_train_global[batch_ind], shift_global)
                    x_train_global_cent[batch_ind] = roll_tensor(x_train_global_cent[batch_ind], shift_global)
                    x_train_local_cent[batch_ind] = roll_tensor(x_train_local_cent[batch_ind], shift_local)
                    

                if 'all' in augment and 'mirror' not in augment:
                    if random() < 0.5:
                        x_train_local[batch_ind] = invert_tensor(x_train_local[batch_ind])
                        x_train_global[batch_ind] = invert_tensor(x_train_global[batch_ind])
                        x_train_local_cent[batch_ind] = invert_tensor(x_train_local_cent[batch_ind])
                        x_train_global_cent[batch_ind] = invert_tensor(x_train_global_cent[batch_ind])
                n_train+=1
            
            ### fix dimensions for next steps
            x_train_local = x_train_local.unsqueeze(1)
            x_train_local_cent = x_train_local_cent.unsqueeze(1)
            x_train_global = x_train_global.unsqueeze(1)
            x_train_global_cent = x_train_global_cent.unsqueeze(1)
            x_train_star = x_train_star.unsqueeze(1)
            #y_train = y_train.unsqueeze(1)

            ### calculate loss using model
            output_train = model(x_train_local, x_train_local_cent, x_train_global, x_train_global_cent, x_train_star)
            #print(output_train.max())
            #print(output_train.min())
            ##assert (output_train.cpu().numpy() >= 0. & output_train.cpu().numpy() <= 1.).all()
            loss = criterion(output_train, y_train)
            train_loss += loss.data

            ### train model (zero gradients and back propogate results)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ### record training loss for this epoch (divided by size of training dataset)
        epoch_train_loss.append(train_loss.cpu().numpy() / len(kepler_data_loader.dataset))
        
        ######################
        ### for validation set
        
        ### loop over batches
        val_pred, val_gt, val_loss, num_corr = np.zeros((3,3)), np.zeros(3), 0, 0
        num_corr_pl=0;    num_corr_ebs=0;    num_corr_unk=0;     #num_corr_bebs=0
        epoch_pl=0;       epoch_ebs=0;       epoch_unk=0;        #epoch_bebs=0;
        epoch_pl_preds=0; epoch_ebs_preds=0; epoch_unk_preds=0;  #epoch_bebs_preds=0;

        for x_val_data, y_val in kepler_val_loader:
                        
            ### get local view, global view, and label for validating
            x_val_local, x_val_global, x_val_local_cent, x_val_global_cent, x_val_star = x_val_data
            x_val_local = Variable(x_val_local).type(torch.FloatTensor).cuda()
            x_val_local_cent=Variable(x_val_local_cent).type(torch.FloatTensor).cuda()
            x_val_global = Variable(x_val_global).type(torch.FloatTensor).cuda()
            x_val_global_cent  = Variable(x_val_global_cent).type(torch.FloatTensor).cuda()
            x_val_star = Variable(x_val_star).type(torch.FloatTensor).cuda()
            y_val = Variable(y_val).type(torch.LongTensor).cuda()
 
            ### fix dimensions for next steps
            x_val_local = x_val_local.unsqueeze(1)
            x_val_local_cent = x_val_local_cent.unsqueeze(1)
            x_val_global = x_val_global.unsqueeze(1)
            x_val_global_cent = x_val_global_cent.unsqueeze(1)
            x_val_star = x_val_star.unsqueeze(1)
            #y_val = y_val.unsqueeze(1)

            ### calculate loss & add to sum over all batches
            output_val = model(x_val_local, x_val_local_cent, x_val_global, x_val_global_cent, x_val_star)
            loss_val = criterion(output_val, y_val)
            val_loss += loss_val.data

            ### get number of correct predictions using threshold=0.5
            ### & sum over all batches
            num_corr += output_val.max(1)[1].eq(y_val).sum().item()
                        
            num_corr_unk += (output_val.max(1)[1].eq(0)*output_val.max(1)[1].eq(y_val)).sum().item();   epoch_unk += (y_val.eq(0)).sum().item();  epoch_unk_preds += (output_val.max(1)[1].eq(0)).sum().item()
            num_corr_pl += (output_val.max(1)[1].eq(1)*output_val.max(1)[1].eq(y_val)).sum().item();   epoch_pl += (y_val.eq(1)).sum().item();  epoch_pl_preds += (output_val.max(1)[1].eq(1)).sum().item()
            num_corr_ebs += (output_val.max(1)[1].eq(2)*output_val.max(1)[1].eq(y_val)).sum().item();  epoch_ebs += (y_val.eq(2)).sum().item();  epoch_ebs_preds += (output_val.max(1)[1].eq(2)).sum().item()
            #num_corr_bebs += (output_val.max(1)[1].eq(3)*output_val.max(1)[1].eq(y_val)).sum().item();  epoch_bebs += (y_val.eq(3)).sum().item();  epoch_bebs_preds += (output_val.max(1)[1].eq(3)).sum().item()
                        
            ### record predictions and ground truth by model
            ### (used for AP per epoch; reset at each epoch; final values output)
            val_pred=np.vstack((val_pred,output_val.data.cpu().numpy()))
            val_gt=np.hstack((val_gt,y_val.data.cpu().numpy()))
        
        val_pred=val_pred[3:]
        val_gt=val_gt[3:]
        
        ### record validation loss calculate for this epoch (divided by size of validation dataset)
        epoch_val_loss.append(val_loss.cpu().numpy() / len(kepler_val_loader.dataset))
        #print("epoch_val_loss",epoch_val_loss," from ",val_loss.cpu().numpy())
        
        ### record validation accuracy (# correct predictions in val set) for this epoch
        epoch_val_recall_pl.append(num_corr_pl/np.max([epoch_pl,0.1]))
        epoch_val_recall_ebs.append(num_corr_ebs/np.max([epoch_ebs,0.1]))
        #epoch_val_recall_bebs.append(num_corr_bebs/np.max([epoch_bebs,0.1]))
        epoch_val_recall_unk.append(num_corr_unk/np.max([epoch_unk,0.1]))
        
        ### record validation accuracy (# correct predictions in val set) for this epoch
        epoch_val_acc_pl.append(num_corr_pl/np.max([epoch_pl_preds,0.1]))
        epoch_val_acc_ebs.append(num_corr_ebs/np.max([epoch_ebs_preds,0.1]))
        #epoch_val_acc_bebs.append(num_corr_bebs/np.max([epoch_bebs_preds,0.1]))
        epoch_val_acc_unk.append(num_corr_unk/np.max([epoch_unk_preds,0.1]))
        
        ### calculate average precision for this epoch
        val_gt=np.hstack(val_gt)
        one_hot_labels=make_one_hot(val_gt,3)
        val_pred=np.vstack(val_pred)
        ### calculate precision and recall curves
        #unk_P, unk_R, _ = precision_recall_curve(one_hot_labels[:,0], val_pred[:,0])
        #pls_P, pls_R, _ = precision_recall_curve(one_hot_labels[:,1], val_pred[:,1])
        #ebs_P, ebs_R, _ = precision_recall_curve(one_hot_labels[:,2], val_pred[:,2])
        #bebs_P, bebs_R = precision_recall_curve(one_hot_labels[:,3], val_pred[:,3])
        epoch_val_ap_pl.append(average_precision_score(one_hot_labels[:,1], val_pred[:,1], average=None))
        epoch_val_ap_unk.append(average_precision_score(one_hot_labels[:,0], val_pred[:,0], average=None))
        epoch_val_ap_ebs.append(average_precision_score(one_hot_labels[:,2], val_pred[:,2], average=None))
        #epoch_val_ap_bebs.append(average_precision_score(one_hot_labels[:,3], val_pred[:,3], average=None))
        epoch_val_ap.append(average_precision_score(one_hot_labels, val_pred, average="micro"))
        
        # Stopping the loop when the val_loss is below 1 (eg not ridiculously high), but has started to increase wrt previous bin of 25:

        if epoch_val_loss[-1]<minloss:
            bests=[val_pred,val_gt]
            minloss=epoch_val_loss[-1]
        ### grab final predictions and ground truths for validation set
        if epoch%5==1:
            tnow=time.time()
            #AP = average_precision_score(np.concatenate(val_gt).ravel(), np.concatenate(val_pred).ravel())
            print(epoch,"{0:.3f}".format((tnow-t0)/epoch-startepoch),"s/epoch , LOSS: train  =  ",str(epoch_train_loss[-1])[:6],
                  ' ,  val  = ',str(epoch_val_loss[-1])[:6],
                  ' , PL acc = '+str(epoch_val_acc_pl[-1])[:6],
                  ' ,  PL recall = '+str(epoch_val_acc_pl[-1])[:6],
                  ' ,  PL AP = '+str(epoch_val_ap_pl[-1])[:6],
                  ' ,  total AP = '+str(epoch_val_ap[-1])[:6])

            dic={}
            dic['epoch']=epoch
            dic['loss_train_epoch']=epoch_train_loss
            dic['loss_val_epoch']=epoch_val_loss
            dic['acc_val_epoch_pl']=epoch_val_acc_pl
            dic['acc_val_epoch_ebs']=epoch_val_acc_ebs
            dic['acc_val_epoch_unk']=epoch_val_acc_unk
            #dic['acc_val_epoch_bebs']=epoch_val_acc_bebs
            dic['recall_val_epoch_pl']=epoch_val_recall_pl
            dic['recall_val_epoch_ebs']=epoch_val_recall_ebs
            dic['recall_val_epoch_unk']=epoch_val_recall_unk
            #dic['recall_val_epoch_bebs']=epoch_val_recall_bebs
            dic['ap_val_epoch']=epoch_val_ap
            dic['ap_val_epoch_pl']=epoch_val_ap_pl
            dic['ap_val_epoch_unk']=epoch_val_ap_unk
            dic['ap_val_epoch_ebs']=epoch_val_ap_ebs
            #dic['ap_val_epoch_bebs']=epoch_val_ap_bebs
            dic['optimizer']=optimizer
            dic['pred_val_final']=bests[0]
            dic['gt_val_final']=bests[1]

            if savefile is not None:
                pickle.dump(dic,open(savefile,'wb'))
                torch.save(model.state_dict(),path.join(foldname,savename+'_temp.pth'))
            #if epoch>75 and np.median(epoch_val_loss[-25:])<1.0 and (np.average(epoch_val_loss[-25:])-np.average(epoch_val_loss[-50:-25]))/np.std(epoch_val_loss[-25:])>1.5:
            #    return dic

    dic={}
    dic['epoch']=epoch
    dic['loss_train_epoch']=epoch_train_loss
    dic['loss_val_epoch']=epoch_val_loss
    dic['acc_val_epoch_pl']=epoch_val_acc_pl
    dic['acc_val_epoch_ebs']=epoch_val_acc_ebs
    dic['acc_val_epoch_unk']=epoch_val_acc_unk
    #dic['acc_val_epoch_bebs']=epoch_val_acc_bebs
    dic['recall_val_epoch_pl']=epoch_val_recall_pl
    dic['recall_val_epoch_ebs']=epoch_val_recall_ebs
    dic['recall_val_epoch_unk']=epoch_val_recall_unk
    #dic['recall_val_epoch_bebs']=epoch_val_recall_bebs
    dic['ap_val_epoch']=epoch_val_ap
    dic['ap_val_epoch_pl']=epoch_val_ap_pl
    dic['ap_val_epoch_unk']=epoch_val_ap_unk
    dic['ap_val_epoch_ebs']=epoch_val_ap_ebs
    #dic['ap_val_epoch_bebs']=epoch_val_ap_bebs
    dic['optimizer']=optimizer
    dic['pred_val_final']=bests[0]
    dic['gt_val_final']=bests[1]

    return dic
#INPUTS:
foldname    = "/home/hosborn/TESS/final_runs/exonet_multiclass3_CV_globcents3b_k8"
savename    = "exonet_CV_4.8_101_all_Big"
savedicname = "exonet_multiclass3_CV_globcents3b_k8_dic"
mod         = "Big"
aug         = "all"
fpath       = "101"
kcount      = "4"
#cont       = False

#Assigning a GPU:
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

model=Model().cuda()
lr = 2.05e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
batch_size = 64
n_epochs = 500

print(savename,"Loading datasets")
kepler_val_data = KeplerDataLoaderCrossVal(infofiles=pickle.load(open(path.join(foldname,savename+'_valfiles.pickle'),'rb')))
kepler_train_data = KeplerDataLoaderCrossVal(infofiles=pickle.load(open(path.join(foldname,savename+'_trainfiles.pickle'),'rb')))
kepler_batch_sampler = pickle.load(open(path.join(foldname,savename+'_BBS.pickle'),'rb'))
kepler_data_loader = DataLoader(kepler_train_data, batch_sampler = kepler_batch_sampler, num_workers=4)
kepler_val_loader = DataLoader(kepler_val_data, batch_size=batch_size, shuffle=False, num_workers=4)

dic=None

print(savename,"starting training")
outputdic = train_model(n_epochs, kepler_data_loader, kepler_val_loader, model, criterion, optimizer,augment="all",savefile=path.join(foldname,savename+'_tempdic.pickle'),cont=dic)
print("saving "+savename)
torch.save(model.state_dict(),path.join(foldname,savename+'.pth'))

outputdic['k']=kcount;outputdic['fpath']=fpath;outputdic['aug']=aug
outputdic['mod']=mod
outputdic['unqid']=fpath+'_'+aug+'_'+mod

pickle.dump(outputdic,open(path.join(foldname,savename+'_final_dic.pickle'),'wb'))
if path.exists(path.join(foldname,"exonet_multiclass3_CV_globcents3b_k8_dic.pickle")):
    outputvals=pickle.load(open(path.join(foldname,"exonet_multiclass3_CV_globcents3b_k8_dic.pickle"),'rb'))
else:
    outputvals={}
outputvals[savename]=outputdic
pickle.dump(outputvals,open(path.join(foldname,"exonet_multiclass3_CV_globcents3b_k8_dic.pickle"),'wb'))
