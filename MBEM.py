import mxnet as mx
import numpy as np
import logging,os
import copy
import urllib
import logging,os,sys
from scipy import stats
from random import shuffle
from __future__ import division

# Downloading data for CIFAR10
# The following function downloads .rec iterator and .lst files (MXNET iterators) for CIFAR10 
# that are used for training the deep learning model with noisy annotations
def download_cifar10():
    fname = ['train.rec', 'train.lst', 'val.rec', 'val.lst']
    testfile = urllib.URLopener()
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_train.rec', fname[0])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_train.lst', fname[1])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_val.rec',   fname[2])
    testfile.retrieve('http://data.mxnet.io/data/cifar10/cifar10_val.lst',   fname[3])
    return fname
# download data
fname = download_cifar10()
# setting up values according to CIFAR10 dataset
# n is total number of training samples for CIFAR10
# n1 is the total number of test samples for CIFAR10 
# k is the number of classes
n, n1, k = 50000, 10000, 10
    
#setting the number of gpus that are available
gpus = None #'0,1,2,3' # if there are no gpus available set it to None.

# m is the number of workers,  gamma is the worker quality, 
# class_wise is the binary variable: takes value 1 if workers are class_wise hammer spammer 
# and 0 if workers are hammer-spammer
# k is the number of classification classes, 
# epochs is the number of epochs for ResNet model
m, gamma, class_wise, epochs, depth  = 100, 0.2, 0, 2, 20

#### main function ####    
def main(fname,n,n1,k,conf,samples,repeat,epochs,depth,gpus):    
    # defining the range of samples that are to be used for training the model
    valid = np.arange(0,samples)
    # declaring the other samples to be invalid 
    invalid = np.arange(samples,n)

    # calling function generate_labels_weight which generates noisy labels given the true labels 
    # the true lables of the examples are ascertained from the .lst files 
    # it takes as input the following:
    # name of the .lst files for the training set and the validation set
    # conf: the confusion matrices of the workers
    # repeat: number of redundant labels that need to be generated for each sample
    # for each i-th sample repeat number of workers are chosen randomly that labels the given sample
    # it returns a multi dimensional array resp_org: 
    # such that resp_org[i,j,k] is 0 vector if the a-th worker was not chosen to label the i-th example
    # else it is one-hot representation of the noisy label given by the j-th worker on the i-th example
    # workers_train_label_org: it is a dictionary. it contains "repeat" number of numpy arrays, each of size (n,k)
    # the arrays have the noisy labels given by the workers
    # workers_val_label: it is a dictionary. it contains one numpy array of size (n,k) 
    # that has true label of the examples in the validation set
    # workers_this_example: it is a numpy array of size (n,repeat).
    # it conatins identity of the worker that are used to generate "repeat" number of noisy labels for example    
    resp_org, workers_train_label_org, workers_val_label, workers_this_example = generate_labels_weight(fname,n,n1,repeat,conf)    
    #setting invalid ones 0, so that they are not used by deep learning module
    for r in range(repeat):
        workers_train_label_org['softmax'+ str(r) +'_label'][invalid] = 0       
    
    print "Algorithm: majority vote:\t\t",
    # running the baseline algorithm where the noisy labels are aggregated using the majority voting
    # calling majority voting function to aggregate the noisy labels
    pred_mv = majority_voting(resp_org[valid])    
    # call_train function takes as input the noisy labels "pred_mv", trains ResNet model for the given "depth"
    # for "epochs" run using the available "gpus". 
    # it prints the generalization error of the trained model.
    _, val_acc = call_train(n,samples,k,pred_mv,workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)
    
    print "Algorithm: weighted majority vote:\t", 
    # running the another baseline algorithm where the aggregation is performed using the weighted majority vote
    # creating a numpy array to store weighted majority vote labels
    naive_agg = np.zeros((n,k))
    # generating the weighted majority vote label using the original noisy labels stored in the 
    # dictionary "workers_train_label_org"
    for r in range(repeat):
        naive_agg = naive_agg + (1/repeat)*copy.deepcopy(workers_train_label_org['softmax'+ str(r) +'_label']) 
    # calling the "call_train" function which besides printing the generalization error 
    # returns model prediction on the training examples, which is being stored in the variable "naive_pred".
    naive_pred, val_acc = call_train(n,samples,k,naive_agg[valid],workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)

    print "Algorithm: MBEM:\t\t\t",    
    # running the proposed algorithm "MBEM: model bootstrapped expectation maximization" 
    # computing posterior probabilities of the true labels given the noisy labels and the worker identities.
    # post_prob_DS function takes the noisy labels given by the workers "resp_org", model prediction obtained 
    # by running "weighted majority vote" algorithm, and the worker identities.
    probs_est_labels = post_prob_DS(resp_org[valid],naive_pred[valid],workers_this_example[valid])      
    algo_agg = np.zeros((n,k))    
    algo_agg[valid] = probs_est_labels
    # calling the "call_train" function with aggregated labels being the posterior probability distribution of the 
    # examples given the model prediction obtained using the "weighted majority vote" algorithm.
    _, val_acc = call_train(n,samples,k,algo_agg[valid],workers_val_label,fname,epochs,depth,gpus)
    print "generalization_acc:  " + str(val_acc)
    
def call_train(n,samples,k,workers_train_label_use,workers_val_label,fname,epochs,depth,gpus):
    # this function takes as input aggregated labels of the training examples
    # along with name of the .rec files for training the ResNet model, depth of the model, number of epochs, and gpus information
    # it returns model prediction on the training examples.
    # we train the model twice first using the given aggregated labels and
    # second using the model prediction on the training examples on based on the first training
    # this aspect is not covered in the algorithm given in the paper. however, it works better in practice.
    # training the model twice in this fashion can be replaced by training once for sufficiently large number of epochs
    
    # first training of the model using the given aggregated labels 
    workers_train_label_use_core = np.zeros((n,k))
    workers_train_label_use_core[np.arange(samples)] = workers_train_label_use        
    pred_first_iter, val_acc = call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus)
    # second training of the model using the model prediction on the training examples based on the first training.
    workers_train_label_use_core = np.zeros((n,k))
    workers_train_label_use_core[np.arange(samples)] = pred_first_iter[np.arange(samples)]
    pred_second_iter, val_acc = call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus)
    return pred_second_iter, val_acc
    
def call_train_core(n,samples,k,workers_train_label_use_core,workers_val_label,fname,epochs,depth,gpus):
    # this function takes as input the same variables as the "call_train" function and it calls
    # the mxnet implementation of ResNet training module function "train" 
    workers_train_label = {} 
    workers_train_label['softmax0_label'] = workers_train_label_use_core  
    prediction, val_acc = train(gpus,fname,workers_train_label,workers_val_label,numepoch=epochs,batch_size=500,depth = depth,lr=0.5)
    model_pred = np.zeros((n,k))
    model_pred[np.arange(samples), np.argmax(prediction[0:samples],1)] = 1
    return model_pred, val_acc 

def generate_workers(m,k,gamma,class_wise):
    # Generating worker confusion matrices according to class-wise hammer-spammer distribution if class_wise ==1
    # Generating worker confusion matrices according to hammer-spammer distribution if class_wise ==0    
    # One row for each true class and columns for given answers
    
    #iniializing confusion matrices with all entries being equal to 1/k that is corresponding to a spammer worker.
    conf = (1/float(k))*np.ones((m,k,k))
    # a loop to generate confusion matrix for each worker 
    for i in range(m): 
        # if class_wise ==0 then generating worker confusion matrix according to hammer-spammer distribution
        if(class_wise==0):
            #letting the confusion matrix to be identity with probability gamma 
            if(np.random.uniform(0,1) < gamma):
                conf[i] = np.identity(k)
            # To avoid numerical issues changing the spammer matrix each element slightly    
            else:
                conf[i] = conf[i] + 0.01*np.identity(k)
                conf[i] = np.divide(conf[i],np.outer(np.sum(conf[i],axis =1),np.ones(k)))        
        else:
            # if class_wise ==1 then generating each class separately according to hammer-spammer distribution    
            for j in range(k):
                # with probability gamma letting the worker to be hammer for the j-th class
                if(np.random.uniform(0,1) < gamma):
                    conf[i,j,:] = 0
                    conf[i,j,j] = 1 
                # otherwise letting the worker to be spammer for the j-th class. 
                # again to avoid numerical issues changing the spammer distribution slighltly 
                # by generating uniform random variable between 0.1 and 0.11
                else:
                    conf[i,j,:] = 1
                    conf[i,j,j] = 1 + np.random.uniform(0.1,0.11)
                    conf[i,j,:] = conf[i,j,:]/np.sum(conf[i,j,:])
    # returining the confusion matrices 
    return conf

def generate_labels_weight(fname,n,n1,repeat,conf):
    # extracting the number of workers and the number of classes from the confusion matrices
    m, k  = conf.shape[0], conf.shape[1]    
    # a numpy array to store true class of the training examples
    class_train = np.zeros((n), dtype = np.int)
    # reading the train.lst file and storing true class of each training example
    with open(fname[1],"r") as f1:
        content = f1.readlines()
    for i in range(n):
        content_lst = content[i].split("\t")
        class_train[i] = int(float(content_lst[1]))
    
    # a dictionary to store noisy labels generated using the worker confusion matrices for each training example  
    workers_train_label = {}
    # the dictionary contains "repeat" number of numpy arrays with keys named "softmax_0_label", where 0 varies
    # each array has the noisy labels for the training examples given by the workers
    for i in range(repeat):
        workers_train_label['softmax' + str(i) + '_label'] = np.zeros((n,k))   
    
    # Generating noisy labels according the worker confusion matrices and the true labels of the examples
    # a variable to store one-hot noisy label, note that each label belongs to one of the k classes
    resp = np.zeros((n,m,k))
    # a variable to store identity of the workers that are assigned to the i-th example
    # note that "repeat" number of workers are randomly chosen from the set of [m] workers and assigned to each example
    workers_this_example = np.zeros((n,repeat),dtype=np.int)
    
    # iterating over each training example
    for i in range(n):
        # randomly selecting "repeat" number of workers for the i-th example
        workers_this_example[i] = np.sort(np.random.choice(m,repeat,replace=False))
        count = 0
        # for each randomly chosen worker generating noisy label according to her confusion matrix and the true label
        for j in workers_this_example[i]:
            # using the row of the confusion matrix corresponding to the true label generating the noisy label
            temp_rand = np.random.multinomial(1,conf[j,class_train[i],:])
            # storing the noisy label in the resp variable 
            resp[i,j,:] = temp_rand
            # storing the noisy label in the dictionary
            workers_train_label['softmax' + str(count) + '_label'][i] = temp_rand
            count = count +1 
            
    # note that in the dictionary each numpy array is of size only (n,k). 
    # The dictionary is passed to the deep learning module
    # however, the resp variable is a numpy array of size (n,m,k).
    # it is used for performing expectation maximization on the noisy labels

    # initializing a dictionary to store one-hot representation of the true labels for the validation set
    workers_val_label = {}
    # the dictionary contains "repeat" number of numpy arrays with keys named "softmax_0_label", where 0 varies
    # each array has the true labels of the examples in the validation set
    workers_val_label['softmax' + str(0) + '_label'] = np.zeros((n1,k))  
    
    # reading the .lst file for the validation set
    content_val_lst = np.genfromtxt(fname[3], delimiter='\t')
    # storing the true labels of the examples in the validation set in the dictionary
    for i in range(n1):
        workers_val_label['softmax' + str(0) + '_label'][i][int(content_val_lst[i,1])] = 1
    
    # returning the noisy responses of the workers stored in the resp numpy array, 
    # the noisy labels stored in the dictionary that is used by the deep learning module
    # the true lables of the examples in the validation set stored in the dictionary
    # identity of the workers that are assigned to th each example in the training set
    return resp, workers_train_label, workers_val_label, workers_this_example

def majority_voting(resp):
    # computes majority voting label
    # ties are broken uniformly at random
    n = resp.shape[0]
    k = resp.shape[2]
    pred_mv = np.zeros((n), dtype = np.int)
    for i in range(n):
        # finding all labels that have got maximum number of votes
        poss_pred = np.where(np.sum(resp[i],0) == np.max(np.sum(resp[i],0)))[0]
        shuffle(poss_pred)
        # choosing a label randomly among all the labels that have got the highest number of votes
        pred_mv[i] = poss_pred[0]   
    pred_mv_vec = np.zeros((n,k))
    # returning one-hot representation of the majority vote label
    pred_mv_vec[np.arange(n), pred_mv] = 1
    return pred_mv_vec

def post_prob_DS(resp_org,e_class,workers_this_example):
    # computes posterior probability distribution of the true label given the noisy labels annotated by the workers
    # and model prediction
    n = resp_org.shape[0]
    m = resp_org.shape[1]
    k = resp_org.shape[2]
    repeat = workers_this_example.shape[1]
    
    temp_class = np.zeros((n,k))
    e_conf = np.zeros((m,k,k))
    temp_conf = np.zeros((m,k,k))
    
    #Estimating confusion matrices of each worker by assuming model prediction "e_class" is the ground truth label
    for i in range(n):
        for j in workers_this_example[i]: #range(m)
            temp_conf[j,:,:] = temp_conf[j,:,:] + np.outer(e_class[i],resp_org[i,j])
    #regularizing confusion matrices to avoid numerical issues
    for j in range(m):  
        for r in range(k):
            if (np.sum(temp_conf[j,r,:]) ==0):
                # assuming worker is spammer for the particular class if there is no estimation for that class for that worker
                temp_conf[j,r,:] = 1/k
            else:
                # assuming there is a non-zero probability of each worker assigning labels for all the classes
                temp_conf[j,r,:][temp_conf[j,r,:]==0] = 1e-10
        e_conf[j,:,:] = np.divide(temp_conf[j,:,:],np.outer(np.sum(temp_conf[j,:,:],axis =1),np.ones(k)))
    # Estimating posterior distribution of the true labels using confusion matrices of the workers and the original
    # noisy labels annotated by the workers
    for i in range(n):
        for j in workers_this_example[i]: 
            if (np.sum(resp_org[i,j]) ==1):
                temp_class[i] = temp_class[i] + np.log(np.dot(e_conf[j,:,:],np.transpose(resp_org[i,j])))
        temp_class[i] = np.exp(temp_class[i])
        temp_class[i] = np.divide(temp_class[i],np.outer(np.sum(temp_class[i]),np.ones(k)))
        e_class[i] = temp_class[i]           
    return e_class

# The following code implements ResNet using MXNET. It is copied from https://github.com/tornadomeet/ResNet/.
def train(gpus,fname,workers_train_label,workers_val_label,numepoch,batch_size,depth = 20,lr=0.5):    
    output_filename = "tr_err.txt"
    model_num = 1
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if os.path.isfile(output_filename):
        os.remove(output_filename)
    hdlr = logging.FileHandler(output_filename)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 

    kv = mx.kvstore.create('device')
    ### training iterator
    train1 = mx.io.ImageRecordIter(
        path_imgrec         = fname[0],
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax0_label',
        data_shape          = (3, 32, 32), 
        batch_size          = batch_size,
        pad                 = 4, 
        fill_value          = 127,  
        rand_crop           = True,
        max_random_scale    = 1.0,  
        min_random_scale    = 1.0, 
        rand_mirror         = True,
        shuffle             = False,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)    
           
    ### Validation iterator
    val1 = mx.io.ImageRecordIter(
        path_imgrec         = fname[2],
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax0_label', 
        batch_size          = batch_size,
        data_shape          = (3, 32, 32), 
        rand_crop           = False,
        rand_mirror         = False,
        pad = 0,
        num_parts           = kv.num_workers,
        part_index          = kv.rank)

    n = workers_train_label['softmax0_label'].shape[0]
    k = workers_train_label['softmax0_label'].shape[1]
    n1 = workers_val_label['softmax0_label'].shape[0]      
    train2 = mx.io.NDArrayIter(np.zeros(n), workers_train_label, batch_size, shuffle = False,)
    train_iter = MultiIter([train1,train2])          
    val2 = mx.io.NDArrayIter(np.zeros(n1), workers_val_label, batch_size = batch_size,shuffle = False,)
    val_iter = MultiIter([val1,val2]) 
        
    if((depth-2)%6 == 0 and depth < 164):
        per_unit = [int((depth-2)/6)]
        filter_list = [16, 16, 32, 64]
        bottle_neck = False
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    units = per_unit*3
    symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=k,data_type="cifar10", 
                    bottle_neck = False, bn_mom=0.9, workspace=512,
                    memonger=False)
    
    devs = mx.cpu() if gpus is None else [mx.gpu(int(i)) for i in gpus.split(',')]
    epoch_size = max(int(n / batch_size / kv.num_workers), 1)
    if not os.path.exists("./model" + str(model_num)):
        os.mkdir("./model" + str(model_num))
    model_prefix = "model"+ str(model_num) + "/resnet-{}-{}-{}".format("cifar10", depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)

    def custom_metric(label,softmax):
        return len(np.where(np.argmax(softmax,1)==np.argmax(label,1))[0])/float(label.shape[0])
    #there is only one softmax layer with respect to which error of all the labels are computed
    output_names = []
    output_names = output_names + ['softmax' + str(0) + '_output']   
    eval_metrics = mx.metric.CustomMetric(custom_metric,name = 'accuracy', output_names=output_names, label_names=workers_train_label.keys())    
       
    model = mx.mod.Module(
        context             = devs,
        symbol              = mx.sym.Group(symbol),
        data_names          = ['data'],
        label_names         = workers_train_label.keys(),#['softmax0_label']
        )
    lr_scheduler = multi_factor_scheduler(0, epoch_size, step=[40, 50], factor=0.1)
    optimizer_params = {
        'learning_rate': lr,
        'momentum' : 0.9,
        'wd' : 0.0001,
        'lr_scheduler': lr_scheduler}
       
    model.fit(
        train_iter,
        eval_data          = val_iter,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(batch_size, 50),
        epoch_end_callback = checkpoint,
        optimizer           = 'nag',
        optimizer_params   = optimizer_params,        
        num_epoch           = numepoch, 
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        )
    
    epoch_max_val_acc, train_acc, val_acc = max_val_epoch(output_filename)
    #print "val-acc: " + str(val_acc) 
    
    # Prediction on Training data
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix,epoch_max_val_acc)
    model = mx.mod.Module(
        context             = devs,
        symbol              = sym,
        data_names          = ['data'], 
        label_names         = workers_train_label.keys(),#['softmax0_label']
        )
    model.bind(for_training=False, data_shapes=train_iter.provide_data, 
         label_shapes=train_iter.provide_label,)
    model.set_params(arg_params, aux_params, allow_missing=True)    

    outputs = model.predict(train_iter)
    if type(outputs) is list:
        return outputs[0].asnumpy(), val_acc
    else:
        return outputs.asnumpy(), val_acc

def max_val_epoch(filename):
    import re
    TR_RE = re.compile('.*?]\sTrain-accuracy=([\d\.]+)')
    VA_RE = re.compile('.*?]\sValidation-accuracy=([\d\.]+)')
    EPOCH_RE = re.compile('Epoch\[(\d+)\] V+?')
    log = open(filename, 'r').read()    
    val_acc = [float(x) for x in VA_RE.findall(log)]
    train_acc = [float(x) for x in TR_RE.findall(log)]
    index_max_val_acc = np.argmax([float(x) for x in VA_RE.findall(log)])
    epoch_max_val_acc = [int(x) for x in EPOCH_RE.findall(log)][index_max_val_acc]
    return epoch_max_val_acc+1, train_acc[index_max_val_acc], val_acc[index_max_val_acc]

class MultiIter(mx.io.DataIter):
    def __init__(self, iter_list):
        self.iters = iter_list 
        #self.batch_size = 500
    def next(self):
        batches = [i.next() for i in self.iters] 
        return mx.io.DataBatch(data=[t for t in batches[0].data],
                         label= [t for t in batches[1].label],pad=0)
    def reset(self):
        for i in self.iters:
            i.reset()
    @property
    def provide_data(self):
        return [t for t in self.iters[0].provide_data]
    @property
    def provide_label(self):
        return [t for t in self.iters[1].provide_label]
    
def multi_factor_scheduler(begin_epoch, epoch_size, step=[40, 50], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None

    

'''
Reproducing paper:
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stage, filter_list, num_class, data_type, bottle_neck=True, bn_mom=0.9, workspace=512, memonger=False):
    """Return ResNet symbol of cifar10 and imagenet
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stage : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_class : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stage)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    if data_type == 'cifar10':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    elif data_type == 'imagenet':
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    else:
         raise ValueError("do not support {} yet".format(data_type))
    for i in range(num_stage):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_class, name='fc1')
    softmax0 = mx.sym.log_softmax(fc1)
    softmax0_output = mx.sym.BlockGrad(data = softmax0,name = 'softmax0')
    loss = [softmax0_output]
    label = mx.sym.Variable(name='softmax0_label')
    ce = -mx.sym.sum(mx.sym.sum(mx.sym.broadcast_mul(softmax0,label),1))
    loss[:] = loss +  [mx.symbol.MakeLoss(ce, normalization='batch')]
    return loss

# calling  function to generate confusion matrices of workers
conf = generate_workers(m,k,gamma,class_wise)  

# calling the main function that takes as input the following:
# name of .rec iterators and .lst files that to operate on,
# worker confusion matrices, 
# number of epochs for running ResNet model, depth of the model,
# number of gpus available on the machine,
# samples: number of samples to be used for training the model,
# repeat: the number of redundant noisy labels to be used for each training example, 
# that are generated using the worker confusion mtrices
# it prints the generalization error of the model on set aside test data
# note that the samples*repeat is approximately same for each pair
# which implies that the total annotation budget is fixed.
for repeat,samples in [[13,4000],[7,7000],[5,10000],[3,17000],[1,50000]]: 
    print "\nnumber of training examples: " + str(samples) + "\t redundancy: " + str(repeat)
    # calling the main function
    main(fname,n,n1,k,conf,samples,repeat,epochs,depth,gpus)
