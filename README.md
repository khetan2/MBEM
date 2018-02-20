# MBEM
This notebook implements MBEM algorithm proposed in the paper [Learning From Noisy Singly-labeled Data](https://openreview.net/forum?id=H1sUHgb0Z) published at ICLR 2018.

Model Bootstrapped Expectation Maximization (MBEM) is a new algorithm for training a deep learning model using noisy data collected from crowdsourcing platforms such as Amazon Mechanical Turk. MBEM outperforms classical crowdsourcing algorithm "majority vote". In this notebook, we run MBEM on CIFAR-10 dataset. We synthetically generate noisy labels given the true labels and using hammer-spammer worker distribution for worker qualities that is explained in the paper. Under the setting when the total annotation budget is fixed, that is we choose whether to collect "1" noisy label for each of the "n" training samples or collect "r" noisy labels for each of the "n/r" training examples.

we show empirically that it is better to choose the former case, that is collect "1" noisy label per example for as many training examples as possible when the total annotation budget is fixed. It takes a few hours to run this notebook and obtain the desired numerical results when using gpus. We use ResNet deep learning model for training a classifier for CIFAR-10. We use the [ResNet](https://github.com/tornadomeet/ResNet/) MXNET implementation.

For running the code call "python MBEM.py". The code requires Python2, [Apache MXNet](https://mxnet.incubator.apache.org/), numpy and scipy packages. 
If a GPU is available, change line 34 in MBEM.py from gpus = None to gpus = '0'. 

## Numerical Results on ImageNet dataset
The ImageNet-1K dataset contains 1.2M training examples and 50K validation examples. 
We divide test set in two parts: 10K for validation and 40K for test. 
Each example belongs to one of the possible 1000 classes. 
We implement our algorithms using a ResNet-18 that achieves top-1 accuracy of 69.5%
and top-5 accuracy of 89% on ground truth labels. 
We use m=1000 simulated workers. 
Although in general, a worker can mislabel an example to one of the 1000 possible classes, 
our simulated workers mislabel 
an example to only one of the 10 possible classes. 
This captures the intuition that even with a larger number of classes, perhaps only a small number are easily confused for each other.
Therefore, each workers' confusion matrix is of size 10 X 10. 
Note that without this assumption, 
there is little hope of estimating a 1000 X 1000 confusion matrix 
for each worker by collecting only approximately 1200 noisy labels from a worker. 
For rest of the settings, please refer to the [Learning From Noisy Singly-labeled Data](https://openreview.net/forum?id=H1sUHgb0Z) paper. 
In the figure below, we fix total annotation budget to be 1.2M 
and vary redundancy from 1 to 9. 
When redundancy is 9, we have only (1.2/9)M training examples,
each labeled by 9 workers. 
MBEM outperforms baselines
in each of the plots, 
achieving the minimum generalization error 
with many singly annotated training examples.

![Figure 1][logo]

[logo]:https://github.com/khetan2/MBEM/blob/master/2_chs.png     
