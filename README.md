# MBEM
This notebook implements MBEM algorithm proposed in the paper [Learning From Noisy Singly-labeled Data] (https://openreview.net/forum?id=H1sUHgb0Z) published at ICLR 2018.

Model Bootstrapped Expectation Maximization (MBEM) is a new algorithm for training a deep learning model using noisy data collected from crowdsourcing platforms such as Amazon Mechanical Turk. MBEM outperforms classical crowdsourcing algorithm "majority vote". In this notebook, we run MBEM on CIFAR-10 dataset. We synthetically generate noisy labels given the true labels and using hammer-spammer worker distribution for worker qualities that is explained in the paper. Under the setting when the total annotation budget is fixed, that is we choose whether to collect "1" noisy label for each of the "n" training samples or collect "r" noisy labels for each of the "n/r" training examples.

we show empirically that it is better to choose the former case, that is collect "1" noisy label per example for as many training examples as possible when the total annotation budget is fixed. It takes a few hours to run this notebook and obtain the desired numerical results when using gpus. We use ResNet deep learning model for training a classifier for CIFAR-10. We use ResNet MXNET implementation given in https://github.com/tornadomeet/ResNet/.
