import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import data.utils as utils
import data.dataset as dataset
import data.data_generator as dg
from optims import Optim
import lr_scheduler as L
import metrics

from models.encoder import DLMAEncoder
from models import top_k_decoder, AttEnc_DecRNN, decoder_rnn
from models.losses import OCDLosses, OrderFreeLosses, CELosses, logit_loss
from models import rescore

import os
import argparse
import time
import json
import collections
import codecs


#config
parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('-config', default='config_rnn.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-notrain', action='store_true',
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

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
print(use_cuda)
#use_cuda = True
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(config.net_seed)

# data
print('loading data...\n')

item_train, label_train, \
    item_test, label_test = dataset.get_data(config)

trainloader = dg.get_loader(item_train, label_train, config.train_batch_size, 
                                    config, shuffle = True, balance = True)
testloader = dg.get_loader(item_test, label_test, config.test_batch_size, 
                                   config, shuffle = False, balance = False)


#metric calculator
train_kinds = set([tuple(x) for x in label_train])
test_kinds = set([tuple(x) for x in label_test])
E = metrics.eval_metrics(train_kinds, test_kinds, config.eval_metrics_split_labels)

all_metrics = metrics.eval_metrics(is_split_label = config.eval_metrics_split_labels).metrics

if config.eval_metrics_split_labels :
    standard_metric = 'overall_micro_f1'
else:
    standard_metric = 'in_train_micro_f1'

scores = [[] for metric in all_metrics]
scores = collections.OrderedDict(zip(all_metrics, scores))
# model
print('building model...\n')
encoder = DLMAEncoder(config.freq_bins, config.label_set_size, config.DLMA_hidden_units, 
                      config.decoder_hidden_size, config.encoder_n_layer)

decoder = decoder_rnn.DecoderRNN(vocab_size = config.label_set_size + 2, max_len = config.decoder_max_len, 
                                 hidden_size = config.decoder_hidden_size, sos_id = config.label_set_size, 
                                 eos_id = config.label_set_size + 1, n_layers = config.decoder_n_layer, 
                                 rnn_cell = 'lstm', input_dropout_p = config.input_dropout_p,
                                 dropout_p = config.dropout_p, use_attention = config.use_attention,
                                 sampling_type=config.decoder_sampling_type, add_mask = config.add_mask)

model = AttEnc_DecRNN.AttEnc_DecRNN(encoder, decoder)

## Loss
if config.decoder_type.lower() == 'ocd':
    Loss = OCDLosses(config.label_set_size + 1, config.OCD_temperature_start,
                        config.OCD_temperature_end, config.OCD_final_hard_epoch)
elif config.decoder_type.lower() == 'vanilla':
    Loss = CELosses(config.label_set_size + 1)
elif config.decoder_type.lower() == 'order_free':
    Loss = OrderFreeLosses(config.label_set_size + 1)

# Restore
if opt.restore:
    model.load_state_dict(checkpoints['model'])

# CUDA
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:  
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
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

# train
def train(epoch):
    model.train()
    model.decoder.set_sampling_type(config.decoder_sampling_type) 
    global updates
    
    if config.schedule:
        scheduler.step()
        print("Decaying learning rate to %g" % scheduler.get_lr()[0])

    if config.decoder_type.lower() == 'vanilla':
        if epoch > config.teacher_forcing_final_epoch:
            teacher_forcing_ratio = config.teacher_forcing_ratio_end 
        else:
            teacher_forcing_ratio = config.teacher_forcing_ratio_start + (config.teacher_forcing_ratio_end - config.teacher_forcing_ratio_start) \
                / config.teacher_forcing_final_epoch * (epoch-1)
        logging("Teacher forcing ratio: " + str(teacher_forcing_ratio) + '\n')
    else:
        teacher_forcing_ratio = 0
    
    total_log_loss, total_rnn_loss, total = 0., 0., 0

    for items, label_set_rnn, label_set_vec in trainloader:
        items = items.float()
        if use_cuda:
            items = items.cuda()
            label_set_rnn = label_set_rnn.cuda()
            label_set_vec = label_set_vec.cuda()
        
        model.zero_grad()

        target_variable = None
        candidates = None
        label_sets = label_set_vec.clone()
        
        if config.decoder_type.lower() == 'vanilla':
            target_variable = label_set_rnn
            label_sets = label_set_rnn
        elif config.decoder_type.lower() == 'order_free':
            candidates = label_set_vec.clone()

        decoder_outputs, decoder_hidden, ret_dict, log_output = model(items, target_variable = target_variable,
                                                                  candidates = candidates,
                                                                  teacher_forcing_ratio=teacher_forcing_ratio)
        rnn_loss = Loss(decoder_outputs, ret_dict['sequence'], label_sets)
        log_loss = logit_loss(log_output, label_set_vec[:,:config.label_set_size]) * config.logistic_weight
        losses = rnn_loss + log_loss
        losses.backward()
        optim.step()
        total_log_loss += log_loss.item()
        total_rnn_loss += rnn_loss.item()
        
        total += 1
        updates += 1

        if updates % 1000 == 0:
            logging(time.strftime("[%H:%M:%S]", time.localtime()))
            logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
            logging("Training loss : {:.5f} \nLog loss : {:.5f}\n".format(total_rnn_loss / total, total_log_loss / total))
    
    logging(time.strftime("[%H:%M:%S]", time.localtime()))
    logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
    logging("Training loss : {:.5f} \nLog loss : {:.5f}\n".format(total_rnn_loss / total, total_log_loss / total))
    
    if config.decoder_type.lower() == 'ocd':
        Loss.update_temperature(epoch)
    
    score =  eval(epoch, 'greedy', config.logistic_joint_decoding)
    score_bs =  eval(epoch, 'beam_search', config.logistic_joint_decoding)

    for metric, value in score.items():
        scores[metric].append(score[metric])
        if metric == standard_metric and score[metric] >= max(scores[metric]):  
            save_model(log_path+'best_'+metric+'_checkpoint.pt')

def eval(epoch, decode_type = 'greedy', logistic_joint_decoding = False):
    #decode_type : greedy or beam_search 
    
    total_rnn_loss, total_log_loss, total = 0.,0., 0.
    
    y_logistic, y_rnn, y_rescore, y  = [], [], [], []

    if decode_type == 'beam_search':
        eval_model = AttEnc_DecRNN.AttEnc_DecRNN(encoder, top_k_decoder.TopKDecoder(model.decoder, config.beam_size))
    elif decode_type == 'greedy':
        model.decoder.set_sampling_type('max') 
        eval_model = model

    eval_model.eval()
    for items, label_set_rnn, label_set_vec in testloader:
        items = items.float()
        if use_cuda:
            items = items.cuda()
            label_set_rnn = label_set_rnn.cuda()
            label_set_vec = label_set_vec.cuda()
        
        decoder_outputs, decoder_hidden, ret_dict, log_output = eval_model(items, logistic_joint_decoding = logistic_joint_decoding)
        
        if config.decoder_type.lower() == 'vanilla':
            label_sets = label_set_rnn
        else:
            label_sets = label_set_vec

        rnn_loss = Loss(decoder_outputs, ret_dict['sequence'], label_sets)
        log_loss = logit_loss(log_output, label_set_vec[:,:config.label_set_size]) * config.logistic_weight
        
        total_log_loss += log_loss.item()
        total_rnn_loss += rnn_loss.item()
        total += 1

        y_vec = E.idx2vec(ret_dict['sequence'], config.label_set_size, config.label_set_size+1 , True)
        y_rnn.append(y_vec)
        y_logistic.append(log_output.detach().cpu().numpy())
        y.append(label_set_vec.cpu().numpy()[:,:config.label_set_size])

        if decode_type == 'beam_search':
            seq, score = rescore.logistic_rescore(ret_dict['topk_sequence'], log_output)
            y_vec = E.idx2vec(seq, config.label_set_size, config.label_set_size +1 , True )
            y_rescore.append(y_vec)

    logging("Test RNN loss : {:.5f}  \nLog loss :{:.5f}\n".format(total_rnn_loss / total, total_log_loss / total))
    
    
    E.set_thres(0.5)
    
    def get_score(y, y_score, typ):
        y_np = np.vstack(y)
        y_score_np = np.vstack(y_score)
        logging("-"*20 + typ + '-'*20 + '\n')
        loss_dict = E.compute(y_np, y_score_np)
        logging(E.logging(loss_dict))
        return loss_dict
    
    loss_d = get_score(y, y_rnn, 'RNN')
    
    get_score(y, y_logistic, 'Logistic')
    if decode_type == 'beam_search':
        get_score(y, y_rescore, 'Logistic Rescore')
    
    logging('-'*50+'\n')
    
    return loss_d

def test(load_checkpoint = False):
    if load_checkpoint:
        checkpoints = torch.load(log_path+'best_{}_checkpoint.pt'.format(standard_metric))
        model.load_state_dict(checkpoints['model'])
    loss_dict = eval(0, 'greedy', config.logistic_joint_decoding)
    loss_dict_bs = eval(0, 'beam_search', config.logistic_joint_decoding)
    return loss_dict

def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)

def main():
    end_epoch = 0
    if opt.notrain:
        test()
        exit()
    for i in range(1, config.epoch+1):
        end_epoch = i
        try:
            train(i)
        except KeyboardInterrupt:
            logging('Interupt\n')
            break
    idx = np.argmax(scores[standard_metric])
    best_epoch = idx+1
    
    logging("Summary (validation):\n")
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
    logging("\nPerformance on test set:\n") 
    test_d = test(True)
    
    d = {}
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
        d[metric] = scores[metric][idx]
    

if __name__ == '__main__':
    main()
