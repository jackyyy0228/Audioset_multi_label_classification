import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import models.dnn
import models.multi_attention
import data.dataloader as dataloader
import data.utils as utils
import data.dataset as dataset
from optims import Optim
import lr_scheduler as L
import metrics
from recorder import Recorder

import os
import argparse
import time
import json
import collections
import codecs


#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-notrain', default=False, type=bool,
                      help="train or not")
parser.add_argument('-result_csv', default='results.csv', type=str,
                    help="recorder path")
opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(config.net_seed)

start_local_time = time.localtime()
print('#Start:',utils.format_time(time.localtime()))

# checkpoint
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']
    config.train_batch_size = config.batch_size
    config.test_batch_size = config.batch_size
    config.decoder_max_len = 17
    config.label_type = 'vec'
    config.label_order = 'freq_first'

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(config.net_seed)

# data
print('loading data...\n')
start_time = time.time()

item_train, label_train, item_test, label_test = dataset.get_data(config)

trainloader = dataloader.get_loader(item_train, label_train, config.train_batch_size, 
                                    config, shuffle = True, balance = True)
validloader = dataloader.get_loader(item_test, label_test, config.test_batch_size, 
                                    config, shuffle = False, balance = False)
testloader = dataloader.get_loader(item_test, label_test, config.test_batch_size, 
                                   config, shuffle = False, balance = False)

#metric calculator
train_kinds = set([tuple(x) for x in label_train])
test_kinds = set([tuple(x) for x in label_test])
#E = metrics.eval_metrics(train_kinds, test_kinds, config.eval_metrics_split_labels)
E = metrics.eval_metrics(train_kinds, test_kinds, False)

# model
print('building model...\n')
model = models.multi_attention.DecisionLevelMultiAttention(config) 

if opt.restore:
    #model.load_state_dict(checkpoints['model'])
    model.load_state_dict(checkpoints['netG'])
    print("sucessfully load the model.")
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:  
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    #optim = checkpoints['optim']
    optim = checkpoints['optimG']
    
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# total number of parameters
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

if not os.path.exists(config.log):
    os.makedirs(config.log)

if config.log.endswith('/'):
    log_path = config.log
else:
    log_path = config.log + '/'

if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt') 

for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  

logging('total number of parameters: %d\n\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

#threshold for binary classifier
threshold = 0.5

total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
all_metrics = metrics.eval_metrics(is_split_label = config.eval_metrics_split_labels).metrics
scores = [[] for metric in all_metrics]
train_loss = []
scores = collections.OrderedDict(zip(all_metrics, scores))

# train
def train(epoch):
    model.train()

    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    l1_crit = nn.L1Loss(size_average=False)
    global updates, total_loss, report_total

    for items, label_sets in trainloader:
        label_sets = label_sets[:,:config.label_set_size]
        if use_cuda:
            items = items.cuda()
            label_sets = label_sets.cuda()
        model.zero_grad()
        outputs  = model(items)
        loss = models.dnn.logit_loss(outputs, label_sets)
        
        if config.reg_factor > 0:
            reg_loss = 0
            for p in model.parameters():
                reg_loss += (p**2).sum()

            loss += config.reg_factor * reg_loss
        loss.backward()

        total_loss += loss.item()
        report_total += 1
        optim.step()
        updates += 1 
        
    logging(time.strftime("[%H:%M:%S]", time.localtime()))
    logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
    logging("Training loss : {:.3f}\n".format(total_loss / report_total))
    
    train_loss.append(total_loss / report_total)
    score = eval(epoch)
    
    for metric, value in score.items():
        scores[metric].append(score[metric])
        if metric == 'overall_auc' and score[metric] >= max(scores[metric]):  
            save_model(log_path+'best_'+metric+'_checkpoint.pt')

    total_loss = 0
    report_total = 0


def eval(epoch, typ = 'Valid'):
    model.eval()
    loader = validloader if typ == 'Valid' else testloader
    
    total_val_loss,val_total = 0.,0
    global threshold
    y_score, y = [], []
    for items, label_sets in loader:
        label_sets = label_sets[:,:config.label_set_size]
        items = items.float()
        label_sets = label_sets.float()
        if use_cuda:
            items = items.cuda()
            label_sets = label_sets.cuda()
        outputs = model(items)
        
        loss = models.dnn.logit_loss(outputs, label_sets)
        total_val_loss += loss.item()
        val_total += 1
        
        y_score.append(outputs.cpu().detach().numpy())
        y.append(label_sets.cpu().numpy())
        
    logging("\n{} loss : {:.3f}\n".format(typ, total_val_loss / val_total))
    
    y_score = np.vstack(y_score)
    y = np.vstack(y)
    y_score.dump(os.path.join(config.log,'epoch{}.prob'.format(epoch)))
    
    _,threshold = E.find_best_thres(y,y_score)
    E.set_thres(threshold)
    
    loss_dict = E.compute(y,y_score)
    
    logging(E.logging(loss_dict))
    logging('-'*50+'\n')
    return loss_dict

def test(epoch):
    global threshold
    checkpoints = torch.load(log_path+'best_overall_auc_checkpoint.pt')
    model.load_state_dict(checkpoints['model'])
    threshold = checkpoints['threshold']
    loss_dict = eval(epoch, 'Test')
    return loss_dict

def save_model(path):
    global updates, threshold
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates,
        'threshold':threshold}
    torch.save(checkpoints, path)

def main():
    end_epoch = 0
    for i in range(1, config.epoch+1):
        end_epoch = i
        try:
            if not opt.notrain:
                train(i)
            else:
                eval(i)
        except KeyboardInterrupt:
            logging('Interupt\n')
            break
    idx = np.argmax(scores['overall_auc'])
    best_epoch = idx+1
    logging("Summary (validation):\n")
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
    logging("\nPerformance on test set:\n") 
    test_d = test(best_epoch)
    
    d = {}
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
        d[metric] = scores[metric][idx]
    
    train_loss_d = {'logit_loss' : train_loss[idx]} 
    
    results = utils.combine_results(start_local_time, time.localtime(), time.time() - start_time, 
                                    best_epoch, 'baseline', end_epoch, threshold, config, train_loss_d, d, test_d)
    
    # Recorder
    recorder = Recorder(opt.result_csv)
    recorder.add_result(results)
    recorder.write_csv()
    print(log_path)

if __name__ == '__main__':
    main()
