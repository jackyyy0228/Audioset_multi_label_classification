import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import models.utils as utils

def logit_loss(outputs, labels):
    losses = labels * torch.log(outputs + 1e-8) + (1-labels) * torch.log(1-outputs + 1e-8)
    losses = - torch.mean(losses)
    return losses 

class OCDLosses(nn.Module):
    def __init__(self, eos_id, init_temp, end_temp, final_epoch):
        self.eos_id = eos_id
        self.temperature = float(init_temp)
        self.init_temp = float(init_temp)
        self.end_temp = float(end_temp)
        self.final_epoch = final_epoch #num of epoch to reduce temperature to zero

    def __call__(self, outputs, output_symbols, targets):
        '''
        Inputs:
            outputs: (seq_len, batch_size, label_size)
            output_symbols : (seq_len, batch_size) index of output symbol (sampling from policy)
            targets: (batch_size, label_size) 
        '''
        # some details:
        # directly minize specific score
        # give sos low score
        outputs = torch.stack(outputs)
        targets = targets.to(outputs.device)
        
        output_symbols = torch.stack(output_symbols).squeeze(2)
        seq_len, batch_size, label_size = outputs.shape

        outputs_one_hot = utils.to_one_hot(output_symbols, label_size).to(outputs.device)
        q_values = torch.zeros(outputs.shape, dtype = torch.float32, device = outputs.device)
        
        mask = torch.ones((seq_len, batch_size), dtype = torch.float32, device = outputs.device)
        
        q_values[0,:,:] = -1 + targets
        for i in range(1, seq_len):
            is_correct = targets * outputs_one_hot[i-1,:,:] # batch_size * label_size
            targets = targets - is_correct
            q_values[i,:,:] = q_values[i-1,:,:] - is_correct + torch.sum(is_correct,dim=1).unsqueeze(1) - 1
            
            # check if all targets are sucessfully predicted
            is_end_batch = torch.sum(targets,dim = 1).eq(0)
            q_values[i,:,self.eos_id] += is_end_batch.float()

            # check eos in output token
            eos_batches = output_symbols[i-1,:].data.eq(self.eos_id)
            eos_batches = eos_batches.float()
            mask[i,:] = (1 - eos_batches) * mask[i-1,:]

        optimal_policy = torch.softmax(q_values / self.temperature, dim = 2)
        #print(F.kl_div(optimal_policy, outputs))
        
        # KL divergence
        #softmax
        #losses =  torch.mean(optimal_policy * torch.log(optimal_policy / (outputs + 1e-8) + 1e-8), dim = 2) * mask
        #log_softmax:
        losses = torch.mean(optimal_policy * (torch.log(optimal_policy+1e-8) -  outputs), dim =2) * mask 
        loss = torch.sum(losses) / torch.sum(mask)
        return loss

    def update_temperature(self, epoch):
        if epoch >= self.final_epoch:
            self.temperature = self.end_temp
        else:
            self.temperature = self.init_temp - (self.init_temp - self.end_temp) / self.final_epoch * float(epoch)

class COTLosses(nn.Module):
    def __init__(self, eos_id):
        ## For OCD target
        super(COTLosses,self).__init__()
        self.eos_id = eos_id
    def __call__(self, outputs, output_symbols, targets):
        outputs = torch.stack(outputs)
        targets = targets.to(outputs.device)
        
        output_symbols = torch.stack(output_symbols).squeeze(2)
        seq_len, batch_size, label_size = outputs.shape

        outputs_one_hot = utils.to_one_hot(output_symbols, label_size).to(outputs.device)
        target_each_time = torch.zeros(outputs.shape, dtype = torch.float32, device = outputs.device)
        
        mask = torch.ones((seq_len, batch_size), dtype = torch.float32, device = outputs.device)
        
        target_each_time[0,:,:] = targets
        for i in range(1, seq_len):
            is_correct = targets * outputs_one_hot[i-1,:,:] # batch_size * label_size
            targets = targets - is_correct
            
            target_each_time[i,:,:] = targets 
            
            # check if all targets are sucessfully predicted
            is_end_batch = torch.sum(targets,dim = 1).eq(0)
            target_each_time[i,:,self.eos_id] += is_end_batch.float()

            # check eos in output token
            eos_batches = output_symbols[i-1,:].data.eq(self.eos_id)
            eos_batches = eos_batches.float()
            mask[i,:] = (1 - eos_batches) * mask[i-1,:]
        
        prob_outputs = torch.exp(outputs)
        new_probs = prob_outputs * (1 - target_each_time) 
        new_probs = new_probs / torch.sum(new_probs, dim = -1 ).unsqueeze(-1)
        Entropy =   torch.sum(new_probs * torch.log(new_probs + 1e-8), dim = -1) * mask
        loss = torch.sum(Entropy) / torch.sum(mask)

        return loss

class CELosses(nn.Module):
    def __init__(self, eos_id):
        super(CELosses, self).__init__()
        self.eos_id = eos_id
        self.criterion = nn.NLLLoss()

    def __call__(self, outputs, output_symbols, label):
        '''
        outputs: (seq_len, batch_size, label_size)
        label: (batch_size, seq_len)
        '''
        outputs = torch.stack(outputs)

        seq_len, batch_size, label_size = outputs.shape
        
        outputs = outputs.transpose(0,1) # batch_size * seq_len * label_size
        outputs = outputs.transpose(1,2) # batch_size * label_size * seq_len
        
        label = label[:,1:] # Don't count sos symbol
        if label.shape[1] != seq_len:
            outputs = outputs[:,:,:-1]
            seq_len = seq_len -1
        
        mask = torch.ones((batch_size, seq_len), dtype = torch.float32, device = outputs.device)
        
        for i in range(1, seq_len):
            # check eos in output token
            eos_batches = label[:,i].data.eq(self.eos_id)
            eos_batches = eos_batches.float()
            mask[:,i] = (1 - eos_batches) * mask[:,i-1]

        losses = self.criterion(outputs, label) * mask
        loss = torch.sum(losses) / torch.sum(mask)
        
        return loss

class OrderFreeLosses(nn.Module):
    def __init__(self, eos_id):
        super(OrderFreeLosses, self).__init__()
        self.eos_id = eos_id
        self.criterion = nn.NLLLoss()

    def __call__(self, outputs, output_symbols, label):
        '''
        Inputs:
            outputs: (seq_len, batch_size, label_size)
            output_symbols : (seq_len, batch_size) index of output symbol (sampling from policy)
        '''
        # some details:
        # directly minize specific score
        # give sos low score
        outputs = torch.stack(outputs)
        
        output_symbols = torch.stack(output_symbols).squeeze(2)
        
        seq_len, batch_size, label_size = outputs.shape
        
        outputs = outputs.transpose(0,1) # batch_size * seq_len * label_size
        outputs = outputs.transpose(1,2) # batch_size * label_size * seq_len
        
        mask = torch.ones((seq_len, batch_size), dtype = torch.float32, device = outputs.device)
        mask[1:,:] = 1 - output_symbols[:-1,:].data.eq(self.eos_id).float() 
        
        losses = self.criterion(outputs, output_symbols.transpose(0,1)) * mask
        loss = torch.sum(losses) / torch.sum(mask)
        
        return loss
        
     
