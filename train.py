import logging
import sys

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.optim import lr_scheduler
from torchvision import models

from model import CompatModel
from utils import AverageMeter, BestSaver, config_logging, prepare_dataloaders

import pickle
import json

from torchmetrics import F1Score


# parser bool type
def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Leave a comment for this training, and it will be used for name suffix of log and saved model
import argparse
parser = argparse.ArgumentParser(description='Fashion Compatibility Training.')
parser.add_argument('--need_rep', type=str2bool, default=True)
parser.add_argument('--vse_off', action="store_true")
parser.add_argument('--pe_off', action="store_true")
parser.add_argument('--sae_off', action="store_true")
parser.add_argument('--sesim_off', action="store_true")

parser.add_argument('--rep2sae', type=str2bool, default=False)
parser.add_argument('--attn2sae', type=str2bool, default=False)

parser.add_argument('--mlp_layers', type=int, default=2)
parser.add_argument('--embed_size', type=int, default=1000)
parser.add_argument('--conv_feats', type=str, default="1234")

parser.add_argument('--num_blocks', type=int, default=1)
parser.add_argument('--self_attention', type=str2bool, default=False)


parser.add_argument('--epochs', type=int, default = 50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default = 1e-2) 
parser.add_argument('--scheduler', type=str, default = None) 

parser.add_argument('--data_name', type=str, default = 'o2f30') 

parser.add_argument('--comment', type=str, default="")
parser.add_argument('--device', type = str, default = 'cuda:0')
parser.add_argument('--seed', type=int, default = None) 

args = parser.parse_args()


print(args)
comment = args.comment
need_rep = args.need_rep
vse_off = args.vse_off
pe_off = args.pe_off
sae_off = args.sae_off
sesim_off = args.sesim_off
mlp_layers = args.mlp_layers
embed_size = args.embed_size
conv_feats = args.conv_feats

epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
device = args.device

# Logger
config_logging(comment)
if args.seed is not None:
    set_seed(args.seed)

# Imgpath_dic
with open('/dataset/imgpath_dic.pkl', 'rb') as f:
    imgpath_dic = pickle.load(f) 

# Dataloader
train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
    prepare_dataloaders(imgpath_dic, args.data_name, batch_size=batch_size)
)


# Model
model = CompatModel(args, embed_size=embed_size, need_rep=need_rep, vocabulary = len(train_dataset.vocabulary), num_style=len(train_dataset.style_dict), 
                    vse_off=vse_off, pe_off=pe_off, sae_off = sae_off, sesim_off = sesim_off, mlp_layers=mlp_layers, conv_feats=conv_feats)




# Train process
# def train(model, device, train_loader, val_loader, comment):
def train(model, device, train_loader, val_loader, args):

    
    model = model.to(device)
    
    criterion = nn.BCELoss()
    
    # Added for optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # MODIFIED for FINAL: 고정된 lr를 args로 변경할 수 있도록 세팅
        
        
    if args.scheduler is not None:
        if args.scheduler.lower() == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    saver = BestSaver(args.comment)
    
    for epoch in range(1, args.epochs + 1):
        logging.info("Train Phase, Epoch: {}".format(epoch))
        total_losses = AverageMeter()
        clf_losses = AverageMeter()
        vse_losses = AverageMeter()
        sae_losses = AverageMeter()
        sclf_losses = AverageMeter()
        sesim_losses = AverageMeter()
        
        # Train phase
        model.train()
        for batch_num, batch in enumerate(train_loader, 1):
            lengths, images, names, texts, offsets, set_ids, labels, is_compat, style = batch # Added for TXT(토큰이 아닌 상품명을 저장하는 texts 추가)
            style = torch.LongTensor(style).to(device)
            is_compat = is_compat.to(device)
            style = style[is_compat==1]
            
            images = images.to(device)
            
            
            # Forward
            output, p_, vse_loss, tmasks_loss, features_loss, sclf_loss, sae_loss, basis_loss, sesim_loss = model(images, texts, names, style, is_compat) # Added for TXT(바뀐 로스 및 파라미터로 설정)
                    
            # BCE Loss
            target = is_compat.float().to(device)
            output = output.squeeze(dim=1)
                                
            clf_loss = criterion(output, target)
            
            # Sum all losses up
            features_loss = 5e-3 * features_loss
            tmasks_loss = 5e-4 * tmasks_loss

            basis_loss = 0.1 * basis_loss

            total_loss = clf_loss + vse_loss + features_loss + tmasks_loss +sclf_loss + sae_loss + basis_loss + sesim_loss 

            # Update Recoder
            total_losses.update(total_loss.item(), images.shape[0])
            clf_losses.update(clf_loss.item(), images.shape[0])
            vse_losses.update(vse_loss.item(), images.shape[0])
            sae_losses.update(sae_loss.item(), images.shape[0])
            sclf_losses.update(sclf_loss.item(), images.shape[0])
            sesim_losses.update(sesim_loss.item(), images.shape[0])
            
            
            # Backpropagation
            optimizer.zero_grad()
            
            total_loss.backward() 
            
            optimizer.step()

            if batch_num % 10 == 0:
                logging.info(
                    "[{}/{}] #{} clf_loss: {:.4f}, vse_loss: {:.4f}, features_loss: {:.4f}, tmasks_loss: {:.4f}, sae_loss: {:.4f}, sclf_loss: {:.4f}, sesim_loss: {:.4f}, basis_loss: {:.4f}, total_loss:{:.4f}".format(
                        epoch, args.epochs, batch_num, clf_losses.val, vse_losses.val, features_loss, tmasks_loss, sae_losses.val, sclf_losses.val, sesim_losses.val, basis_loss, total_losses.val
                    )
                )
        logging.info("Train Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        
       
        # scheduler
        if args.scheduler is not None:
            scheduler.step()
        
        
        # Valid Phase
        logging.info("Valid Phase, Epoch: {}".format(epoch))
        model.eval()
        
        clf_losses = AverageMeter()
        sae_losses = AverageMeter()
        
        f1 = F1Score(num_classes = len(train_dataset.style_dict), average='weighted')
        # compat
        compat_outputs = []
        compat_targets = []
        # style
        style_probs = []
        style_truth = []
        for batch_num, batch in enumerate(val_loader, 1):
            lengths, images, names, texts, offsets, set_ids, labels, is_compat, style = batch 

            style = torch.LongTensor(style).to(device)
            is_compat = is_compat.to(device)
            style = style[is_compat==1]

            images = images.to(device)
            target = is_compat.float().to(device)
            with torch.no_grad():
                if need_rep:
                    output, features, _, rep  = model._compute_multiattention_score(images, texts)
                else:
                    output, features, _ =  model._compute_multiattention_score(images, texts)
                
                if sae_off==False:
                    if model.args.rep2sae == True:
                        p_, sae_loss, _ = model._compute_rep2style_loss(rep, is_compat)
                    if model.args.attn2sae == True:
                        p_, sae_loss, _ = model._compute_attn2style_loss(features, is_compat)   
                        
                if sesim_off == False:
                    p_, _, _ = model._compute_sesim_loss(rep, style, is_compat)
                    
                    
                output = output.squeeze(dim=1)
                clf_loss = criterion(output, target)            
                
            clf_losses.update(clf_loss.item(), images.shape[0])
            sae_losses.update(sae_loss.item(), images.shape[0])
            
            
            compat_outputs.append(output)
            compat_targets.append(target)
            
            if ((model.sae_off)==False) or ((model.sesim_off)==False):
                style_probs.append(p_)
                style_truth.append(style)
        
        logging.info("Valid Loss (clf_loss): {:.4f}".format(clf_losses.avg))
        
        
        ### AUC (Compatibility learning)
        compat_outputs = torch.cat(compat_outputs).cpu().data.numpy()
        compat_targets = torch.cat(compat_targets).cpu().data.numpy()
        auc = metrics.roc_auc_score(compat_targets, compat_outputs)
        logging.info("AUC: {:.4f}".format(auc))
       
        ### F1(Style classification)
        if ((model.sae_off)==False) or ((model.sesim_off)==False):
            style_probs = torch.cat(style_probs).cpu()
            style_truth = torch.cat(style_truth).cpu()
            f1_score = f1(style_probs, style_truth)
            logging.info("F1: {:.4f}".format(f1_score))
        
        ### Average ((F1+AUC)/2)
        logging.info("Average:{:.4f}".format((auc+f1_score)/2))
        
        predicts = np.where(compat_outputs > 0.5, 1, 0)
        accuracy = metrics.accuracy_score(predicts, compat_targets)
        logging.info("Accuracy@0.5: {:.4f}".format(accuracy))
        positive_loss = -np.log(compat_outputs[compat_targets==1]).mean()
        logging.info("Positive loss: {:.4f}".format(positive_loss))
        positive_acc = sum(compat_outputs[compat_targets==1]>0.5) / len(compat_outputs)
        logging.info("Positive accuracy: {:.4f}".format(positive_acc))
        
        # Added for Early Stop
        if (auc+f1_score)/2 > saver.best:
            patient = 0
            
        else:
            patient += 1
            
        
        # Save best model
        saver.save((auc+f1_score)/2, model.state_dict())
        
        
        with open('args/{}.txt'.format(args.comment), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        if patient > 15:
            break
            

if __name__ == "__main__":
    train(model, device, train_loader, val_loader, args)
