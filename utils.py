import logging

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset

from dataset import CategoryDataset, collate_fn


class AverageMeter(object):
    """Computes and stores the average and current value.

    >>> acc = AverageMeter()
    >>> acc.update(0.6)
    >>> acc.update(0.8)
    >>> print(acc.avg)
    0.7
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BestSaver(object):
    """Save pytorch model with best performance. Template for save_path is:

        model_{save_path}_{comment}.pth

    >>> comment = "v1"
    >>> saver = BestSaver(comment)
    >>> auc = 0.6
    >>> saver.save(0.6, model.state_dict())
    """
    def __init__(self, comment=None):
        # Get current executing script name
        import __main__, os
        exe_fname=os.path.basename(__main__.__file__)
        save_path = "model_{}".format(exe_fname.split(".")[0])
        
        if comment is not None and str(comment):
#             save_path = save_path + "_" + str(comment)
            save_path = str(comment)
        save_path = save_path + ".pth"

        self.save_path = save_path
        self.best = float('-inf')

    def save(self, metric, data):
        if metric > self.best:
            self.best = metric
            torch.save(data, "save/"+self.save_path)
            logging.info("Saved best model to {}".format(self.save_path))


def config_logging(comment=None):
    """Configure logging for training log. The format is 

        `log_{log_fname}_{comment}.log`

    .g. for `train.py`, the log_fname is `log_train.log`.
    Use `logging.info(...)` to record running log.

    Args:
        comment (any): Append comment for log_fname
    """

    # Get current executing script name
    import __main__, os
    exe_fname=os.path.basename(__main__.__file__)
    log_fname = "log_{}".format(exe_fname.split(".")[0])

    if comment is not None and str(comment):
#         log_fname = log_fname + ":" + str(comment)
        log_fname = str(comment)
    
    log_fname = "log/" + log_fname + ".log"
    log_format = "%(asctime)s [%(levelname)-5.5s] %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.FileHandler(log_fname), logging.StreamHandler()]
    )

def prepare_dataloaders(imgpath_dic, data_name, root_dir="../data/images/", data_dir="/paper_dataset/", batch_size=16, img_size=224, use_mean_img=True, neg_samples=True, num_workers=0, collate_fn=collate_fn):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
        ]
    )
    
    data_dic = data_collection(data_name)
    
    train_dataset = CategoryDataset(
        imgpath_dic,
        word_dic=data_dic['wd'],
        attr_dic=data_dic['ad'],
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file=data_dic['tr'],
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )
    val_dataset = CategoryDataset(
        imgpath_dic,
        word_dic=data_dic['wd'],
        attr_dic=data_dic['ad'],
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file=data_dic['val'],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    test_dataset = CategoryDataset(
        imgpath_dic,
        word_dic=data_dic['wd'],
        attr_dic=data_dic['ad'],
        root_dir=root_dir,
        data_dir=data_dir,
        transform=transform,
        use_mean_img=use_mean_img,
        data_file=data_dic['te'],
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader


def data_collection(name):
    '''
    o - outfit 1/2
    f - frequency 10/20/30/40/50
    a - attribute
    '''
    if name == 'o2f10':
        data_dic = {'tr': 'tr_outfit2.json', 'val': 'val_outfit2.json', 'te': 'te_outfit2.json', 'wd': 'musinsa_word_dict_10.txt', 'ad': None}
    elif name == 'o2f20':
        data_dic = {'tr': 'tr_outfit2.json', 'val': 'val_outfit2.json', 'te': 'te_outfit2.json', 'wd': 'musinsa_word_dict_20.txt', 'ad': None}
    elif name == 'o2f30':
        data_dic = {'tr': 'tr_outfit2.json', 'val': 'val_outfit2.json', 'te': 'te_outfit2.json', 'wd': 'musinsa_word_dict_30.txt', 'ad': None}
    elif name == 'o2f40':
        data_dic = {'tr': 'tr_outfit2.json', 'val': 'val_outfit2.json', 'te': 'te_outfit2.json', 'wd': 'musinsa_word_dict_40.txt', 'ad': None}
    elif name == 'o2f50':
        data_dic = {'tr': 'tr_outfit2.json', 'val': 'val_outfit2.json', 'te': 'te_outfit2.json', 'wd': 'musinsa_word_dict_50.txt', 'ad': None}
    elif name == 'o2f10_a':
        data_dic = {'tr': 'tr_outfit2_attr.json', 'val': 'val_outfit2_attr.json', 'te': 'te_outfit2_attr.json', 'wd': 'musinsa_word_dict_10.txt', 'ad': 'musinsa_attr_dict_30.txt'}
    elif name == 'o2f20_a':
        data_dic = {'tr': 'tr_outfit2_attr.json', 'val': 'val_outfit2_attr.json', 'te': 'te_outfit2_attr.json', 'wd': 'musinsa_word_dict_20.txt', 'ad': 'musinsa_attr_dict_30.txt'}
    elif name == 'o2f30_a':
        data_dic = {'tr': 'tr_outfit2_attr.json', 'val': 'val_outfit2_attr.json', 'te': 'te_outfit2_attr.json', 'wd': 'musinsa_word_dict_30.txt', 'ad': 'musinsa_attr_dict_30.txt'}
    elif name == 'o2f40_a':
        data_dic = {'tr': 'tr_outfit2_attr.json', 'val': 'val_outfit2_attr.json', 'te': 'te_outfit2_attr.json', 'wd': 'musinsa_word_dict_40.txt', 'ad': 'musinsa_attr_dict_30.txt'}
    elif name == 'o2f50_a':
        data_dic = {'tr': 'tr_outfit2_attr.json', 'val': 'val_outfit2_attr.json', 'te': 'te_outfit2_attr.json', 'wd': 'musinsa_word_dict_50.txt', 'ad': 'musinsa_attr_dict_30.txt'}
    
    return data_dic