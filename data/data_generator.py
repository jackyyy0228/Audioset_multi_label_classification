import numpy as np
import torch 
import math

def get_loader(item, label, batch_size, config, shuffle, balance = True, negative = False):
     if balance :
         dg = BalancedDataGenerator
     else:
         dg = VanillaDataGenerator
     return dg(item, label, batch_size, config, shuffle, negative = negative)

class VanillaDataGenerator(object):
    def __init__(self, x, y, batch_size, config, shuffle=True, negative = False, seed=1234):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.label_set_size = config.label_set_size + 2
        self.seq_len = config.decoder_max_len
        self.sos_id = config.label_set_size 
        self.eos_id = config.label_set_size + 1
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        self.negative = negative
        self.config = config

        if config.label_order == 'freq_first':
            self.y = sort_by_freq(self.y)
        elif config.label_order == 'rare_first':
            self.y = sort_by_freq(self.y, reverse = False)

    def __iter__(self, max_iteration=None):
        
        batch_size = self.batch_size
        
        samples_num = len(self.x)
        indexes = np.arange(samples_num)
        max_iteration = samples_num // batch_size + 1
        
        if self.shuffle:
            self.rs.shuffle(indexes)
        
        iteration = 0
        pointer = 0
        
        while True:
            
            if iteration == max_iteration:
                break
            
            # Get batch indexes
            start = pointer
            if pointer + batch_size >= samples_num:
                end = samples_num
                pointer = 0
                if self.shuffle:
                    self.rs.shuffle(indexes)
            else:
                end = pointer + batch_size

            batch_idxes = indexes[start : end]
            pointer += batch_size

            iteration += 1
            feature = self.x[batch_idxes]
            
            # negative 
            if self.negative:
                neg_batch_idxes = []
                for i,b_idx in enumerate(batch_idxes):
                    flag = True
                    while flag:
                        index2 = int(torch.randint(0, len(self.y), (1,)))
                        if not np.array_equal(self.y[b_idx], self.y[index2]):
                            flag = False
                            neg_batch_idxes.append(index2)
                batch_idxes = np.array(neg_batch_idxes)
            
            batch_labels = []
            for i, b_idx in enumerate(batch_idxes):
                batch_labels.append(list(self.y[b_idx]))

            labels_rnn = trans_batch_label_set(batch_labels, self.seq_len, self.label_set_size, 
                                           self.eos_id, self.sos_id, typ = 'rnn')   
            
            labels_vec = trans_batch_label_set(batch_labels, self.seq_len, self.label_set_size, 
                                           self.eos_id, self.sos_id, typ = 'vec')   
            
            
            yield torch.FloatTensor(feature), labels_rnn, labels_vec

class BalancedDataGenerator(object):
    """Balanced data generator. Each mini-batch is balanced with approximately 
    the same number of samples from each class. 
    """
    
    def __init__(self, x, y, batch_size, config, shuffle=True, negative = False, seed = 1234):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.label_set_size = config.label_set_size + 2
        self.seq_len = config.decoder_max_len
        self.sos_id = config.label_set_size 
        self.eos_id = config.label_set_size + 1
        self.shuffle = shuffle
        self.rs = np.random.RandomState(seed)
        self.negative = negative
        self.config = config

        if config.label_order == 'freq_first':
            self.y = sort_by_freq(self.y)
        elif config.label_order == 'rare_first':
            self.y = sort_by_freq(self.y, reverse = False)
        
            
    def get_classes_set(self, samples_num_of_classes):
        
        classes_num = len(samples_num_of_classes)
        classes_set = []
        
        for k in range(classes_num):
            classes_set += [k]
            
        return classes_set

    def __iter__(self, max_iteration=None):
        y = self.y
        batch_size = self.batch_size

        #(samples_num, classes_num) = y.shape
        samples_num = len(y)
        classes_num = self.label_set_size
        
        max_iteration = samples_num // batch_size 
        
        #samples_num_of_classes = np.sum(y, axis=0)
        samples_num_of_classes = [0] * classes_num
        
        for label_set in y:
            for x in label_set:
                samples_num_of_classes[x] += 1
        
        # E.g. [0, 1, 1, 2, ..., K, K]
        classes_set = [x for x in range(classes_num)]
        
        # E.g. [[0, 1, 2], [3, 4, 5, 6], [7, 8], ...]
        indexes_of_classes = [ [] for _ in range(classes_num)]
        for idx, label_set in enumerate(y):
            for x in label_set:
                indexes_of_classes[x].append(idx)
            
        # Shuffle indexes
        if self.shuffle:
            for k in range(classes_num):
                self.rs.shuffle(indexes_of_classes[k])
        
        queue = []
        iteration = 0
        pointers_of_classes = [0] * classes_num

        while True:
            
            if iteration == max_iteration:
                break
            
            # Get a batch containing classes from a queue
            while len(queue) < batch_size:
                self.rs.shuffle(classes_set)
                queue += classes_set
                
            batch_classes = queue[0 : batch_size]
            queue = []
            
            samples_num_of_classes_in_batch = [batch_classes.count(k) for k in range(classes_num)]
            batch_idxes = []
            
            # Get index of data from each class
            for k in range(classes_num):
                
                bgn_pointer = pointers_of_classes[k]
                fin_pointer = pointers_of_classes[k] + samples_num_of_classes_in_batch[k]
                
                per_class_batch_idxes = indexes_of_classes[k][bgn_pointer : fin_pointer]
                batch_idxes.append(per_class_batch_idxes)

                pointers_of_classes[k] += samples_num_of_classes_in_batch[k]
                
                if pointers_of_classes[k] >= samples_num_of_classes[k]:
                    pointers_of_classes[k] = 0
                    
                    if self.shuffle:
                        self.rs.shuffle(indexes_of_classes[k])
                
            batch_idxes = np.int32(np.concatenate(batch_idxes, axis=0))
            iteration += 1
            
            feature = self.x[batch_idxes]
            
            # negative 
            if self.negative:
                neg_batch_idxes = []
                for i,b_idx in enumerate(batch_idxes):
                    flag = True
                    while flag:
                        index2 = int(torch.randint(0, len(self.y), (1,)))
                        if not np.array_equal(self.y[b_idx], self.y[index2]):
                            flag = False
                            neg_batch_idxes.append(index2)
                batch_idxes = np.array(neg_batch_idxes)

            batch_labels = []
            for i, b_idx in enumerate(batch_idxes):
                batch_labels.append(list(self.y[b_idx]))

            labels_rnn = trans_batch_label_set(batch_labels, self.seq_len, self.label_set_size, 
                                           self.eos_id, self.sos_id, typ = 'rnn')   
            
            labels_vec = trans_batch_label_set(batch_labels, self.seq_len, self.label_set_size, 
                                           self.eos_id, self.sos_id, typ = 'vec')   
            
            
            yield torch.FloatTensor(feature), labels_rnn, labels_vec

def trans_batch_label_set(batch_label_set, seq_len, label_set_size, 
                          eos_id, sos_id, typ = 'index' ):
    '''
    batch_label_set: list of list of labels
    typ : index : the same
          rnn : torch.tensor(batch_size, seq_len) of correspinding symols (vanilla)
          vec : (batch_size, label_set_size) a multi-hot vector for each label set  (OCD,OrderFree)
    '''
    if typ  == 'index' :
        labels = batch_label_set
    elif typ == 'vec':
        labels = np.zeros([len(batch_label_set), label_set_size], dtype=np.float32)
        for i, label_set in enumerate(batch_label_set):
            for x in label_set:
                labels[i][x] = 1
        labels = torch.FloatTensor(labels)
    elif typ == 'rnn':
        labels = np.zeros([len(batch_label_set), seq_len])
        for idx, label_set in enumerate(batch_label_set):
            labels[idx][0] = sos_id
            for t, x in enumerate(label_set):
                labels[idx][t+1] = x
            for t in range(len(label_set) + 1, seq_len):
                labels[idx][t] = eos_id
        labels = torch.LongTensor(labels)
    return labels

def sort_by_freq(label_sets ,reverse = True):
    '''
    label_sets: list of list [[1,3,429], [2,4,194]....]
    reverse : set it False if rare first
    '''
    d = {}
    for label_set in label_sets:
        for x in label_set:
            if x not in d:
                d[x] = 1
            else:
                d[x] += 1

    L = []
    for label_set in label_sets:
        tmp = [ (x, d[x]) for x in label_set]
        tmp = sorted(tmp, reverse = reverse, key = lambda x:x[1] )
        L.append([ x[0] for x in tmp])
    return L
        

if __name__ == '__main__':
    
    x = np.ones((1000, 784))
    y = np.ones((1000, 10))
    
    gen = BalancedDataGenerator(x, y, batch_size=128, shuffle=True, seed=1234)
    
    for (batch_x, batch_y) in gen.generate(max_iteration=3):
        print(batch_x.shape)
