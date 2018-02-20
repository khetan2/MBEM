from __future__ import division
import mxnet as mx
import numpy as np
import logging,os
import copy
import urllib
import logging,os,sys
from scipy import stats
from random import shuffle

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
