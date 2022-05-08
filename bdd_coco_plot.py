import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *

recall_level_default = 0.95


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--energy', type=int, default=1, help='noise for Odin')
parser.add_argument('--T', default=1., type=float, help='temperature: energy|Odin')
parser.add_argument('--thres', default=1., type=float)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()



concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()



# ID data
id_data = pickle.load(open('./data/BDD-Detection/faster-rcnn/'+args.name+'/random_seed' +'_'+str(args.seed)  +'/inference/bdd_custom_val/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))
ood_data = pickle.load(open('./data/BDD-Detection/faster-rcnn/'+args.name+'/random_seed' +'_'+str(args.seed)  +'/inference/coco_ood_val_bdd/standard_nms/corruption_level_0/probabilistic_scoring_res_odd_'+str(args.thres)+'.pkl', 'rb'))

id = 0
T = 1
id_score = []
ood_score = []

assert len(id_data['inter_feat'][0]) == 11 #+ 1024

if args.energy:
    #id_score = -torch.stack(id_data['logistic_score'])
    #ood_score = -torch.stack(ood_data['logistic_score'])
    id_score = -args.T * torch.logsumexp(torch.stack(id_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
    ood_score = -args.T * torch.logsumexp(torch.stack(ood_data['inter_feat'])[:, :-1] / args.T, dim=1).cpu().data.numpy()
else:
    id_score = -np.max(F.softmax(torch.stack(id_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)
    ood_score = -np.max(F.softmax(torch.stack(ood_data['inter_feat'])[:, :-1], dim=1).cpu().data.numpy(), axis=1)



###########
########
print(len(id_score))
print(len(ood_score))

measures = get_measures(-id_score, -ood_score, plot=False)

if args.energy:
    print_measures(measures[0], measures[1], measures[2], 'energy')
else:
    print_measures(measures[0], measures[1], measures[2], 'msp')

