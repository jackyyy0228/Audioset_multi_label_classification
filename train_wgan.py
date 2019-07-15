import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import numpy as np

import models.wgan
import models.multi_attention
import data.dataloader as dataloader
import data.dataset as dataset
import data.utils as utils
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
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-pretrain', default=False, type=bool,
                    help="load pretrain embedding")
parser.add_argument('-notrain', default=False, type=bool,
                    help="train or not")
parser.add_argument('-result_csv', default='results.csv', type=str,
                    help="recorder path")
opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

start_local_time = time.localtime()
print('#Start:',utils.format_time(time.localtime()))

# checkpoint
if opt.restore: 
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config.train_batch_size = config.batch_size
    config.test_batch_size = config.batch_size
    config.decoder_max_len = 17
    config.label_type = 'vec'
    config.label_order = 'freq_first'

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0

if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print(use_cuda)


# data
print('loading data...\n')
start_time = time.time()
item_train, label_train, \
    item_test, label_test = dataset.get_data(config)
'''
trainset = dataset.Dataset(item_train, label_train, config.label_set_size, input_type = config.module_type) 
validset = dataset.Dataset(item_test, label_test, config.label_set_size, input_type = config.module_type) 
testset = dataset.Dataset(item_test, label_test, config.label_set_size, input_type = config.module_type)
'''
trainloader = dataloader.wgan_data_loader(item_train, label_train, config.batch_size, 
                                           config, shuffle = True, balance = True)
validloader = dataloader.wgan_data_loader(item_test, label_test, config.batch_size, 
                                           config, shuffle=False, balance = False)
testloader = dataloader.wgan_data_loader(item_test, label_test, config.batch_size, 
                                          config, shuffle=False, balance = False)

print('loading time cost: %.3f' % (time.time()-start_time))

#metric calculator
train_kinds = set([tuple(x) for x in label_train])
test_kinds = set([tuple(x) for x in label_test])
E = metrics.eval_metrics(train_kinds, test_kinds, config.eval_metrics_split_labels)

print('building model...\n')
#netG = models.dnn.DNNModel(config)
netG = models.multi_attention.DecisionLevelMultiAttention(config)
netD = models.wgan.Discriminator(config, use_cuda) 

if opt.restore:
    netG.load_state_dict(checkpoints['netG'])
    netD.load_state_dict(checkpoints['netD'])
    print("Sucessfully load the netG.\n")

if config.load_feature_extractor:
    netD.load_feature_extractor(config.feature_extractor_path,config.feature_extractor_name)
    print("Sucessfully load the pretrained feature extractor.\n")

## GPUs
if use_cuda:
    netG.cuda()
    netD.cuda()

if len(opt.gpus) > 1:  
    netG = nn.DataParallel(netG, device_ids=opt.gpus, dim=1)
    netD = nn.DataParallel(netD, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optimG = checkpoints['optimG']
    optimD = checkpoints['optimD']
else:
    optimG = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    optimD = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
    
optimG.set_parameters(netG.parameters())
optimD.set_parameters(netD.parameters())

if config.schedule:
    schedulerG = L.CosineAnnealingLR(optimG.optimizer, T_max=config.epoch)
    schedulerD = L.CosineAnnealingLR(optimD.optimizer, T_max=config.epoch)

# total number of parameters

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
    logging("%s: %s\n" % (str(k), str(v)))
logging("\n")

param_count = 0
for param in netG.parameters():
    param_count += param.view(-1).size()[0]
logging('number of parameters of generator: %d\n\n' % param_count)

param_count = 0
for param in netD.parameters():
    param_count += param.view(-1).size()[0]
logging('number of parameters of discriminatorr: %d\n\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

#threshold for binary classifier
threshold = 0.5

# Losses
train_loss_printer = models.wgan.WGANLossPrinter()
valid_loss_printer = models.wgan.WGANLossPrinter()
test_loss_printer = models.wgan.WGANLossPrinter()


# Metrics 
all_metrics = metrics.eval_metrics(is_split_label = config.eval_metrics_split_labels).metrics
scores = [[] for metric in all_metrics]
scores = collections.OrderedDict(zip(all_metrics, scores))


wgan_loss = models.wgan.WGAN_loss(config, netG, netD, use_cuda)
logging("{}\n".format(config.log))

# train
def train(epoch):
    netG.train()
    netD.train()
    
    if config.schedule:
        schedulerG.step()
        schedulerD.step()
        print("Decaying learning rate of generator and disciriminator to %g" % schedulerG.get_lr()[0])

    global updates, total_loss, start_time, report_total
    
    for pos_batch, gen_batch, neg_batch in trainloader:
        netG.zero_grad()
        netD.zero_grad()
        g_losses, d_losses = 0, 0
        
        gen_g_loss, logit_loss, gen_d_loss, kl_loss_gen, g_outputs = wgan_loss.generate_batch(gen_batch)
        g_losses += gen_g_loss
        d_losses += gen_d_loss / 2
        
        if updates % (config.num_d_update + config.num_g_update) < config.num_d_update:
            g_losses += config.alpha * logit_loss
        
            g_losses.backward()
            train_loss_printer.update_g(logit_loss, gen_g_loss, g_losses)
            optimG.step()
        else:
            pos_d_loss, kl_loss_pos = wgan_loss.positive_batch(pos_batch)
            neg_d_loss = wgan_loss.negative_batch(neg_batch)
            gp_loss = netD.gradient_penalty(pos_batch, gen_batch)
            d_losses += pos_d_loss + neg_d_loss / 2 + config.gp_lambda * gp_loss
            
            # KL loss in VDB   
            total_kl_loss = (kl_loss_gen + kl_loss_pos) / 2   
            
            if config.use_vdb:
                d_losses += netD.vdb_beta * total_kl_loss
                netD.update_beta(total_kl_loss.item(), float(config.beta_step_size))
            
            d_losses.backward()
            train_loss_printer.update_d(gen_d_loss , pos_d_loss, neg_d_loss , gp_loss, total_kl_loss, d_losses)
            optimD.step()
        updates += 1  

    logging(time.strftime("[%H:%M:%S]", time.localtime()))
    logging(" Epoch: %3d, updates: %8d\n" % (epoch, updates))
    logging("{}\n".format(config.log))
    logging("Training loss : \n")
    logging(train_loss_printer.logging() + '\n')
    score = eval(epoch,'Valid')
    
    for metric, value in score.items():
        scores[metric].append(score[metric])
        if metric == 'overall_auc' and score[metric] >= max(scores[metric]):  
            save_model(log_path+'best_' + metric + '_checkpoint.pt')
    
    train_loss_printer.save_and_reset(epoch)
    #save_model(log_path+'checkpoint.pt')
    
    return train_loss_printer.check_diverge(epoch)

def eval(epoch, typ = 'Valid'):
    netG.eval()
    netD.eval()

    global threshold
    y_score, y = [], []
    loss_printer = valid_loss_printer if typ == 'Valid' else test_loss_printer
    loader = validloader if typ == 'Valid' else testloader

    for pos_batch, gen_batch, neg_batch in loader:
        g_losses, d_losses = 0,0
        
        gen_g_loss, logit_loss, gen_d_loss, kl_loss_gen, g_outputs = wgan_loss.generate_batch(gen_batch)
        g_losses += gen_g_loss
        d_losses += gen_d_loss / 2

        g_losses += config.alpha * logit_loss
        
        loss_printer.update_g(logit_loss, gen_g_loss, g_losses)
        
        pos_d_loss, kl_loss_pos = wgan_loss.positive_batch(pos_batch)
        neg_d_loss = wgan_loss.negative_batch(neg_batch)
        gp_loss = netD.gradient_penalty(pos_batch, gen_batch)
        d_losses += pos_d_loss + neg_d_loss / 2 + config.gp_lambda * gp_loss
        
        # KL loss
        total_kl_loss = (kl_loss_gen + kl_loss_pos) / 2   
        
        loss_printer.update_d(gen_d_loss , pos_d_loss, neg_d_loss, gp_loss, total_kl_loss, d_losses)
        
        y_score.append(g_outputs.cpu().detach().numpy())
        y.append(gen_batch[1].cpu().numpy())
    
    logging("{} loss : \n".format(typ))
    logging(loss_printer.logging() + '\n')
    loss_printer.save_and_reset(epoch)

    y_score = np.vstack(y_score)
    y = np.vstack(y)
    y_score.dump(os.path.join(config.log,'epoch{}.prob'.format(epoch)))

    if typ == 'Valid':
        _,threshold = E.find_best_thres(y,y_score)
    E.set_thres(threshold)
    
    loss_dict = E.compute(y,y_score)
    
    logging(E.logging(loss_dict))
    logging('-'*50+'\n')
    return loss_dict

def test(epoch):
    global threshold
    checkpoints = torch.load(log_path+'best_overall_micro_f1_checkpoint.pt')
    netG.load_state_dict(checkpoints['netG'])
    netD.load_state_dict(checkpoints['netD'])
    threshold = checkpoints['threshold']
    loss_dict = eval(epoch, 'Test')
    return loss_dict

def save_model(path):
    global updates, threshold
    netG_state_dict = netG.module.state_dict() if len(opt.gpus) > 1 else netG.state_dict()
    netD_state_dict = netD.module.state_dict() if len(opt.gpus) > 1 else netD.state_dict()
    checkpoints = {
        'netG': netG_state_dict,
        'netD': netD_state_dict,
        'config': config,
        'optimG': optimG,
        'optimD': optimD,
        'updates': updates,
        'threshold':threshold}
    torch.save(checkpoints, path)

def main():
    end_epoch = 0
    for i in range(1, config.epoch+1):
        end_epoch = i
        try:
            if not opt.notrain:
                flag = train(i)
                if flag:
                    break
            else:
                eval(i,'Valid')
        except KeyboardInterrupt:
            logging('Interupt\n')
            break
    #Select the best micro_f1 score
    idx = np.argmax(scores['overall_micro_f1'])
    best_epoch = idx+1

    logging("Summary (validation):\n")
    d = {}
    for metric in all_metrics:
        logging("{}:{:.3f}\n".format(metric,scores[metric][idx]))
        d[metric] = scores[metric][idx]

    test_d = test(best_epoch)

    val_losses = utils.combine_dict(valid_loss_printer.losses[best_epoch],d)
    test_losses = utils.combine_dict(test_loss_printer.losses[best_epoch],test_d) 
    results = utils.combine_results(start_local_time, time.localtime(), time.time() - start_time, 
                                    best_epoch, 'wgan', end_epoch, threshold, config, 
                                    train_loss_printer.losses[best_epoch], val_losses, test_losses)
    
    # Recorder
    recorder = Recorder(opt.result_csv)
    recorder.add_result(results)
    recorder.write_csv()
    print(log_path)

if __name__ == '__main__':
    main()
