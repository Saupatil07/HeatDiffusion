import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from sklearn import preprocessing
import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=32, help='image size')
parser.add_argument('--gt', type=str, default='/home/ssp2/Heat_Diffusion/64/gt/', help='path to ground truth data')
parser.add_argument('--cond', type=str, default='/home/ssp2/Heat_Diffusion/64/cond/', help='path to conditions data')

args = parser.parse_args()

def cond_dataload():
   dir_name_cond=  args.cond
   files_cond= sorted(filter(os.path.isfile,glob.glob(dir_name_cond+'*.csv')))   
   print("Total Files:- ",len(files_cond))
   train_cond = files_cond[:int(0.9*len(files_cond))]
   train_cond_dataset = []
   for f in tqdm(train_cond):
      train_cond_dataset.append(np.loadtxt(f,delimiter=","))
   train_cond_dataset = np.array(train_cond_dataset)
   dfmax, dfmin = train_cond_dataset.max(), train_cond_dataset.min()
   print(dfmax,dfmin)
   train_cond_normalized = 2*((train_cond_dataset - dfmin)/(dfmax - dfmin))-1
   np.save("./train_cond_norm.npy",train_cond_normalized)
   
   test_cond = files_cond[int(0.9*len(files_cond)):]
   test_cond_dataset = []
   for f in tqdm(test_cond):
      test_cond_dataset.append(np.loadtxt(f,delimiter=","))
   test_cond_dataset = np.array(test_cond_dataset)
   test_dfmax, test_dfmin = test_cond_dataset.max(), test_cond_dataset.min()
   print(test_dfmax,test_dfmin)
   test_cond_normalized = 2*((test_cond_dataset - test_dfmin)/(test_dfmax - test_dfmin))-1
   np.save("./test_cond_norm.npy",test_cond_normalized)

def gt_dataload():
   dir_name_gt=  args.gt
   files_gt= sorted(filter(os.path.isfile,glob.glob(dir_name_gt+'*.csv')))   
   print("Total Files:- ",len(files_gt))
   train_cond = files_gt[:int(0.9*len(files_gt))]
   train_cond_dataset = []
   for f in tqdm(train_cond):
      train_cond_dataset.append(np.loadtxt(f,delimiter=","))
   train_cond_dataset = np.array(train_cond_dataset)
   dfmax, dfmin = train_cond_dataset.max(), train_cond_dataset.min()
   print(dfmax,dfmin)
   train_cond_normalized = 2*((train_cond_dataset - dfmin)/(dfmax - dfmin))-1
   np.save("./train_gt_norm.npy",train_cond_normalized)
   
   test_cond = files_gt[int(0.9*len(files_gt)):]
   test_cond_dataset = []
   for f in tqdm(test_cond):
      test_cond_dataset.append(np.loadtxt(f,delimiter=","))
   test_cond_dataset = np.array(test_cond_dataset)
   test_dfmax, test_dfmin = test_cond_dataset.max(), test_cond_dataset.min()
   print(test_dfmax,test_dfmin)
   test_cond_normalized = 2*((test_cond_dataset - test_dfmin)/(test_dfmax - test_dfmin))-1
   np.save("./test_gt_norm.npy",test_cond_normalized)

cond_dataload()
gt_dataload()