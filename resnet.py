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

