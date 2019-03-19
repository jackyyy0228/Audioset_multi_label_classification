import numpy as np
import torch
from scipy import stats
from sklearn.metrics import average_precision_score,roc_auc_score


EPS=0.0000001
def score_to_pred(y_score, threshold, top_k):
    if top_k < len(y_score[0]):
        ## If not in top_k, set to 0
        last = len(y_score[0]) - top_k
        y_score_temp = []
        for vec in y_score:
            vec2 = np.array(vec)
            index = vec2.argsort()[:last]
            vec2[index] = 0
            y_score_temp.append(vec2)
        y_score_temp = np.array(y_score_temp)
    else:
        y_score_temp = y_score
    y_pred = np.where(y_score_temp > threshold,1,0)
    return y_pred
    
def cal_f1(y, y_score, threshold = 0.5, top_k = 200):
    # Shape : n_image, n_class
    n_image,n_class = y.shape
    TP = np.zeros(n_class)
    FP = np.zeros(n_class)
    FN = np.zeros(n_class)
    if top_k < len(y_score[0]):
        ## If not in top_k, set to 0
        last = len(y_score[0]) - top_k
        y_score_temp = []
        for vec in y_score:
            vec2 = np.array(vec)
            index = vec2.argsort()[:last]
            vec2[index] = 0
            y_score_temp.append(vec2)
        y_score_temp = np.array(y_score_temp)
    else:
        y_score_temp = y_score
    y_head = score_to_pred(y_score,threshold,top_k)
    for i in range(n_class):
        TP[i] = np.sum(np.where( (y[:,i]+y_head[:,i])/2 == np.ones(n_image),1,0))
        FP[i] = np.sum(y_head[:,i]) - TP[i]
        FN[i] = np.sum(y[:,i]) - TP[i]
    return TP,FP,FN

def macro_f1(y, y_score, threshold = 0.5,top_k = 200):
    n_image,n_class = y.shape
    TP,FP,FN = cal_f1(y, y_score,threshold,top_k)
    precisions = [TP[i]/(TP[i] + FP[i]+EPS) for i in range(n_class)]
    recalls = [TP[i]/(TP[i] + FN[i]+EPS) for i in range(n_class)]
    p = np.mean(precisions)
    r = np.mean(recalls)
    return float(p),float(r),float(2*p*r/(p+r+EPS))
    
def micro_f1(y, y_score,threshold = 0.5,top_k = 200):
    n_image,n_class = y.shape
    TP,FP,FN = cal_f1(y, y_score,threshold,top_k)
    p = np.sum(TP) / ( np.sum(TP) + np.sum(FP) + EPS )
    r = np.sum(TP) / ( np.sum(TP) + np.sum(FN) + EPS )
    return float(p),float(r),float(2*p*r/(p+r+EPS))

def hamming_loss(y, y_score, threshold = 0.5, top_k = 200):
    y_pred = score_to_pred(y_score, threshold, top_k)
    corrects = np.where(y == y_pred,1,0)
    return 1 - float(np.sum(corrects)) / len(y.flatten())

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def cal_auc(y, y_score):
    n_image,n_class = y.shape
    aucs = []
    for i in range(n_class):
        if np.sum(y[:,i]) > 0:
            aucs.append(roc_auc_score(y[:,i], y_score[:,i]))
    return np.mean(aucs)

def cal_map(y, y_score):
    n_image,n_class = y.shape
    aps = []
    for i in range(n_class):
        if np.sum(y[:,i]) > 0:
            aps.append(average_precision_score(y[:,i], y_score[:,i]))
    return np.mean(aps)

def check_zero_class(y):
    n_image,n_class = y.shape
    x = 0
    for i in range(n_class):
        if np.sum(y[:,i]) == 0:
            x += 1
    return x


class eval_metrics():
    def __init__(self, train_kinds = [], test_kinds = [], is_split_label = False):
        self.thres = 0.5
        self.top_k = 200
        metrics = ['hamming_loss','macro_f1','macro_precision',
                        'macro_recall','micro_f1','micro_precision','micro_recall','map','auc','d_prime','n_zero_class']
        self.metrics = [ 'in_train_' + x for x in metrics ]
        self.train_kinds = set([tuple(x) for x in train_kinds])
        self.test_kinds = set([tuple(x) for x in test_kinds])
        self.is_split_label = is_split_label

        if self.is_split_label :
            self.in_train_kinds = self.test_kinds.intersection(self.train_kinds)
            self.out_train_kinds =  self.test_kinds.difference(self.train_kinds)
            self.metrics += [ 'out_train_' + x for x in metrics ]
            self.metrics += [ 'overall_' + x for x in metrics ]
    
    def find_best_thres(self,y, y_score):
        thres_step = 0.1
        best_thres = 0.
        best_score = 0. 
        y, y_score = np.array(y), np.array(y_score)
        for i in range(9):
            thres = thres_step * (i + 1)
            p,r,f1 = micro_f1(y, y_score, threshold=thres)
            if best_score < f1:
                best_score = f1
                best_thres = thres
        return best_score, best_thres

    def set_thres(self,thres):
        self.thres = thres

    def split_labels(self, y, y_score):
        y_in_train, y_out_train = [],[]
        y_score_in_train, y_score_out_train = [],[]

        for label_set, score_set in zip(y,y_score):
            key = transform_label(label_set)
            if key in self.in_train_kinds:
                y_in_train.append(label_set)
                y_score_in_train.append(score_set)
            elif key in self.out_train_kinds:
                y_out_train.append(label_set)
                y_score_out_train.append(score_set)
            #else:
            #    print("There is an error.")
        y_in_train, y_out_train = np.vstack(y_in_train), np.vstack(y_out_train)
        y_score_in_train, y_score_out_train = np.vstack(y_score_in_train), np.vstack(y_score_out_train)
        return y_in_train, y_score_in_train,  y_out_train, y_score_out_train

    def compute(self, y, y_score):
        if self.is_split_label:
            y_in_train, y_score_in_train,  y_out_train, y_score_out_train = self.split_labels(y, y_score)
            d_in_train = self._compute(y_in_train, y_score_in_train)
            d_out_train = self._compute(y_out_train, y_score_out_train)
            d_overall = self._compute(y, y_score)
            d = { 'in_train_' + k:v for k,v in d_in_train.items()}
            for k,v in d_out_train.items():
                d['out_train_' + k] = v
            for k,v in d_overall.items():
                d['overall_' + k] = v
        else:
            d = self._compute(y, y_score)
            d = { 'in_train_' + k:v for k,v in d.items()}
        return d

    def logging(self, loss_dict):
        log = ''
        if self.is_split_label:
            log += 'Label sets appearing in training set: {} / {}\n'.format(len(self.in_train_kinds),len(self.test_kinds))
            log += self._logging(loss_dict, 'in_train')
            log += '\nLabel sets not appearing in training set:{} / {}\n'.format(len(self.out_train_kinds),len(self.test_kinds))
            log += self._logging(loss_dict, 'out_train')
            log += '\nOverall:\n'
            log += self._logging(loss_dict, 'overall')
        else:
            log += self._logging(loss_dict,'in_train')
        return log

    def _compute(self, y, y_score):
        h_loss = hamming_loss(y, y_score, self.thres, self.top_k)
        macro_p, macro_r, macro_f1_score = macro_f1(y, y_score, self.thres, self.top_k)
        micro_p, micro_r, micro_f1_score = micro_f1(y, y_score, self.thres, self.top_k)
        map_score = cal_map(y, y_score)
        auc  = cal_auc(y, y_score)
        dprime = d_prime(auc) 
        n_zero_class = check_zero_class(y)
        d =  {'hamming_loss': h_loss, 
                'macro_f1': macro_f1_score,
                'macro_precision': macro_p, 
                'macro_recall': macro_r,
                'micro_f1': micro_f1_score,
                'micro_precision': micro_p, 
                'micro_recall': micro_r,
                'map': map_score,
                'auc': auc,
                'd_prime': dprime,
                'n_zero_class':n_zero_class}
        return d

    def _logging(self, loss_dict, typ):
        d = loss_dict
        log=""
        log += "threshold: {:.3f}\n".format(self.thres)
        log += "Hamming_loss: {:.4f}\n".format(d[typ + '_hamming_loss'])
        log += "Macro precision, recall, f1: {:.3f}, {:.3f}, {:.3f}\n".format(d[typ + '_macro_precision'],
                                                                      d[typ + '_macro_recall'],d[typ + '_macro_f1'])
        log += "Micro precision, recall, f1: {:.3f}, {:.3f}, {:.3f}\n".format(d[typ + '_micro_precision'],
                                                                      d[typ + '_micro_recall'],d[typ + '_micro_f1'])
        log += "Number of zero class:{}\n".format(d[typ + '_n_zero_class'])
        log += "Map:{:.3f}\n".format(d[typ + '_map'])
        log += "auc:{:.3f}\n".format(d[typ + '_auc'])
        log += "d_prime:{:.3f}\n".format(d[typ + '_d_prime'])
        return log
    def idx2vec(self, y, label_set_size, eos_idx, transpose = False):
        if transpose:
           y = torch.stack(y).squeeze().cpu().numpy().transpose()
        n_sample = len(y)
        labels = np.zeros([n_sample, label_set_size])
        for idx, label_set in enumerate(y):
            for x in label_set:
                if x == eos_idx:
                    break
                if x < label_set_size:
                    labels[idx][x] = 1
        return labels
        
def transform_label(label_set):
    # Transform label set from [0,0,1,1,...1...0] to (3,4,...10..)
    return tuple(np.where(label_set == 1)[0])
    
