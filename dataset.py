import csv
import gzip
import itertools
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torchvision
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import pickle


class CategoryDataset(Dataset):
    """Dataset for polyvore with 5 categories(upper, bottom, shoe, bag, accessory),
    each suit has at least 3 items, the missing items will use a mean image.

    Args:
        root_dir: Directory stores source images
        data_file: A json file stores each outfit index and description
        data_dir: Directory stores data_file and mean_images
        transform: Operations to transform original images to fix size
        use_mean_img: Whether to use mean images to fill the blank part
        neg_samples: Whether generate negative sampled outfits
    """
    def __init__(self,
                 imgpath_dic,
                 word_dic,
                 attr_dic,
                 root_dir="../data/images/",
                 data_file='train_dict_mss.json',
                 data_dir="/paper_dataset/",
                 transform=None,
                 use_mean_img=True,
                 neg_samples=True,
                 ):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.transform = transform
        self.use_mean_img = use_mean_img
        self.data = json.load(open(os.path.join(data_dir, data_file)))
        self.data = [(k, v) for k, v in self.data.items()]
        self.neg_samples = neg_samples # if True, will randomly generate negative outfit samples
        self.imgpath_dic = imgpath_dic
        
        # Style dict: 데이터에 맞게 변경
        if 'outfit2' in data_file:
            with open('/paper_dataset/style_dict2.pkl', 'rb') as f:
                self.style_dict = pickle.load(f)
        else: 
            with open('/paper_dataset/style_dict.pkl', 'rb') as f:
                self.style_dict = pickle.load(f)
                
        self.vocabulary, self.word_to_idx = [], {}
        self.word_to_idx['UNK'] = len(self.vocabulary)
        self.vocabulary.append('UNK')
        self.vocabulary.append('PAD')
        ## ADDED
        self.word_to_idx['upper_mean'] = len(self.vocabulary)
        self.vocabulary.append('upper_mean')
        self.word_to_idx['bottom_mean'] = len(self.vocabulary)
        self.vocabulary.append('bottom_mean')
        self.word_to_idx['shoe_mean'] = len(self.vocabulary)
        self.vocabulary.append('shoe_mean')
        self.word_to_idx['bag_mean'] = len(self.vocabulary)
        self.vocabulary.append('bag_mean')
        
        with open(os.path.join(self.data_dir, word_dic)) as f:
            for line in f:
                name = line.strip().split()[0]
                if name not in self.word_to_idx:
                    self.word_to_idx[name] = len(self.vocabulary)
                    self.vocabulary.append(name)
                    
        ### Attribute 상품명 추가(e.g., 주요소재_면(코튼),부츠타입_워커부츠)
        if attr_dic != None:
            with open(os.path.join(self.data_dir, attr_dic)) as f:
                for line in f:
                    name = line.strip().split()[0]
                    if name not in self.word_to_idx:
                        self.word_to_idx[name] = len(self.vocabulary)
                        self.vocabulary.append(name)



    def __getitem__(self, index):
        """It could return a positive suits or negative suits"""
        set_id, parts = self.data[index]
        
        style = self.style_dict[parts['style']]
        

        if random.randint(0, 1) and self.neg_samples:
            to_change = list(parts.keys()) # random choose negative items
        else:
            to_change = []
        imgs = []
        labels = []
        names = []
        texts = [] 

        '''
        to_change의 목적을 알수 없기에 이 파트의 목적성이 불분명함.
        '''
        for part in ['upper', 'bottom', 'shoe', 'bag']:
            if part in to_change: # random choose a image from dataset with same category
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = self.imgpath_dic[choice[1][part]['index']]
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
                texts.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name']))) #Added for BOW
            elif part in parts.keys(): #이제 이파트는 무의미한 부분.
                img_path = self.imgpath_dic[parts[part]['index']]
                names.append(torch.LongTensor(self.str_to_idx(parts[part]['name'])))
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
                texts.append(torch.LongTensor(self.str_to_idx(parts[part]['name']))) #Added for BOW
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                names.append(torch.LongTensor([])) # mean_img embedding
                labels.append('{}_{}'.format(part, 'mean'))
                texts.append(torch.LongTensor([self.word_to_idx['{}_{}'.format(part, 'mean')]])) #Added for BOW
            else:
                choice = self.data[index]
                while (choice[0] == set_id) or (part not in choice[1].keys()):
                    choice = random.choice(self.data)
                img_path = self.imgpath_dic[choice[1][part]['index']]
                names.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
                labels.append('{}_{}'.format(choice[0], choice[1][part]['index']))
                texts.append(torch.LongTensor(self.str_to_idx(choice[1][part]['name'])))
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            imgs.append(img)
        input_images = torch.stack(imgs)
        is_compat = (len(to_change)==0) # 'True' = 1, 1이 진짜 outfit
        
        offsets = list(itertools.accumulate([0] + [len(n) for n in names[:-1]]))
        offsets = torch.LongTensor(offsets)

        return input_images, names, texts, offsets, set_id, labels, is_compat, style

    def __len__(self):
        return len(self.data)

    def str_to_idx(self, name):
        return [self.word_to_idx[w] if w in self.word_to_idx else self.word_to_idx['UNK']
            for w in name.split()]

    def get_fitb_quesiton(self, index):
        set_id, parts = self.data[index]
        del(parts['style'])
        question_part = random.choice(list(parts))
        question_id = "{}_{}".format(set_id, parts[question_part]['index'])
        imgs = []
        item_texts = [] #Added for TXT
        labels = []
        
        for part in ['upper', 'bottom', 'shoe', 'bag']:
            if part in parts.keys():
                img_path = self.imgpath_dic[parts[part]['index']]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                item_texts.append(torch.LongTensor(self.str_to_idx(parts[part]['name']))) #Added for BoW
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                item_texts.append(torch.LongTensor([self.word_to_idx['{}_{}'.format(part, 'mean')]])) #Added for BoW
                labels.append('{}_{}'.format(part, 'mean'))
        items = torch.stack(imgs)

        option_ids = [set_id]
        options = []
        option_texts = [] #Added for TXT
        option_labels = []
        while len(option_ids) < 4:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (question_part not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = self.imgpath_dic[option[1][question_part]['index']]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_texts.append(torch.LongTensor(self.str_to_idx(option[1][question_part]['name'])))
                option_labels.append("{}_{}".format(option[0], option[1][question_part]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, item_texts, labels, question_part, question_id, options, option_texts, option_labels
    
    def get_generation(self, index, question_part):
        random.seed(index)
        
        """Generate fill in th blank questions.
        Return:
            images: 5 parts of a outfit
            labels: store if this item is empty
            question_part: which part to be changed
            options: 3 other item with the same category,
                expect original composition get highest score
        """
        set_id, parts = self.data[index]
        question_part = question_part #random.choice(['upper', 'bottom'])#random.choice(list(parts))
        question_id = "{}_{}".format(set_id, parts[question_part]['index'])
        imgs = []
        labels = []
        for part in ['upper', 'bottom', 'shoe', 'bag']:#['upper', 'bottom', 'shoe', 'bag']:
            if part in ['upper', 'bottom']:
                img_path = self.imgpath_dic[parts[part]['index']]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}'.format(set_id, parts[part]['index']))
            elif self.use_mean_img:
                img_path = os.path.join(self.data_dir, part+'.png')
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                imgs.append(img)
                labels.append('{}_{}'.format(part, 'mean'))
        items = torch.stack(imgs)

        option_ids = [set_id]
        options = []
        option_labels = []
        while len(option_ids) < 100:
            option = random.choice(self.data)
            if (option[0] in option_ids) or (question_part not in option[1]):
                continue
            else:
                option_ids.append(option[0])
                img_path = self.imgpath_dic[option[1][question_part]['index']]
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
                options.append(img)
                option_labels.append("{}_{}".format(option[0], option[1][question_part]['index']))

        # Return 4 options for question, 3 incorrect options
        return items, labels, question_part, question_id, options, option_labels

def collate_fn(data):
    """Need custom a collate_fn"""
    data.sort(key=lambda x:x[0].shape[0], reverse=True)
    images,  names, texts, offsets, set_ids, labels, is_compat, style = zip(*data) #Added for TXT
    lengths = [i.shape[0] for i in images]
    is_compat = torch.LongTensor(is_compat)
    names = sum(names, [])
    texts = sum(texts, [])
    offsets = list(offsets)
    images = torch.stack(images)
    return (
        lengths,
        images,
        names,
        texts, 
        offsets,
        set_ids,
        labels,
        is_compat,
        style
    )


# Test the loader
if __name__ == "__main__":
    img_size = 224
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor()
    ])
    d = CategoryDataset(transform=transform, use_mean_img=True)
    loader = DataLoader(d, 4, shuffle=True, num_workers=4, collate_fn=collate_fn)
