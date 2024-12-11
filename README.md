Deep Boltzmann Machines
=======================

Deep Boltzmann Machines [1,2] evaluated on the MNIST [3] handwritten digit recognition task; Python+Numpy implementation.

Network: 3 stacked Restricted Boltzmann Machines (RBMs): 784-500, 500-500, 500-2000 followed by 10
softmax outputs for the digits 0-9.

Train on 50k samples, test on 10k; Error rate: 0.0% (train) / 1.45% (test)

See [FF repo](https://github.com/jesper-olsen/ff-py) for another result on the same task.

References:
-----------
[1] [Deep Boltzmann Machines](https://proceedings.mlr.press/v5/salakhutdinov09a.html) <br/>
[2] [Matlab code](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html) <br/>
[3] [LeCun's raw MNIST db](http://yann.lecun.com/exdb/mnist/) <br/>

Run:
----

Download MNIST:   
```
% mkdir -p MNIST/raw
% cd MNIST/raw
% wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
% gunzip *.gz
% cd ../..
```

Train a model:
```
% time python deep_classify.py
Batchsize: 100 Input-dim: 784 #training batches: 500
Pretraining Layer 1 with RBM: 784-500
ep  1/10 error 813482.4
ep  2/10 error 493331.4
ep  3/10 error 431145.0
ep  4/10 error 403328.0
ep  5/10 error 387591.0
ep  6/10 error 413434.0
ep  7/10 error 390326.2
ep  8/10 error 376399.8
ep  9/10 error 367271.1
ep 10/10 error 360456.8
Pretraining Layer 2 with RBM: 500-500
ep  1/10 error 915513.3
ep  2/10 error 601278.4
ep  3/10 error 544336.3
ep  4/10 error 508340.8
ep  5/10 error 479548.9
ep  6/10 error 469781.1
ep  7/10 error 411936.6
ep  8/10 error 386192.6
ep  9/10 error 374881.9
ep 10/10 error 367595.5
Pretraining Layer 3 with RBM: 500-2000
ep  1/10 error 3671708.1
ep  2/10 error 2243720.6
ep  3/10 error 1603298.0
ep  4/10 error 1202086.3
ep  5/10 error 944251.2
ep  6/10 error 624413.0
ep  7/10 error 458412.6
ep  8/10 error 420705.0
ep  9/10 error 403323.1
ep 10/10 error 391466.5
Model size - #weights:  1665010

Training discriminative model on MNIST by minimizing cross entropy error.
ep   1/100 misclassified 44900/50000 (train)  8990/10000 (test)
ep   2/100 misclassified  5510/50000 (train)   966/10000 (test)
ep   3/100 misclassified  4788/50000 (train)   850/10000 (test)
ep   4/100 misclassified  4368/50000 (train)   775/10000 (test)
ep   5/100 misclassified  4105/50000 (train)   724/10000 (test)
ep   6/100 misclassified  3911/50000 (train)   704/10000 (test)
ep   7/100 misclassified  3090/50000 (train)   545/10000 (test)
ep   8/100 misclassified  2604/50000 (train)   475/10000 (test)
ep   9/100 misclassified  2277/50000 (train)   425/10000 (test)
ep  10/100 misclassified  1974/50000 (train)   374/10000 (test)
ep  11/100 misclassified  1829/50000 (train)   366/10000 (test)
ep  12/100 misclassified  1588/50000 (train)   337/10000 (test)
ep  13/100 misclassified  1387/50000 (train)   304/10000 (test)
ep  14/100 misclassified  1238/50000 (train)   275/10000 (test)
ep  15/100 misclassified  1134/50000 (train)   264/10000 (test)
ep  16/100 misclassified  1068/50000 (train)   246/10000 (test)
ep  17/100 misclassified   901/50000 (train)   224/10000 (test)
ep  18/100 misclassified   849/50000 (train)   228/10000 (test)
ep  19/100 misclassified   728/50000 (train)   208/10000 (test)
ep  20/100 misclassified   656/50000 (train)   206/10000 (test)
ep  21/100 misclassified   638/50000 (train)   216/10000 (test)
ep  22/100 misclassified   518/50000 (train)   210/10000 (test)
ep  23/100 misclassified   489/50000 (train)   205/10000 (test)
ep  24/100 misclassified   441/50000 (train)   205/10000 (test)
ep  25/100 misclassified   363/50000 (train)   198/10000 (test)
ep  26/100 misclassified   390/50000 (train)   209/10000 (test)
ep  27/100 misclassified   268/50000 (train)   192/10000 (test)
ep  28/100 misclassified   253/50000 (train)   185/10000 (test)
ep  29/100 misclassified   216/50000 (train)   190/10000 (test)
ep  30/100 misclassified   210/50000 (train)   194/10000 (test)
ep  31/100 misclassified   156/50000 (train)   187/10000 (test)
ep  32/100 misclassified   124/50000 (train)   180/10000 (test)
ep  33/100 misclassified   110/50000 (train)   181/10000 (test)
ep  34/100 misclassified   110/50000 (train)   179/10000 (test)
ep  35/100 misclassified    71/50000 (train)   175/10000 (test)
ep  36/100 misclassified    58/50000 (train)   171/10000 (test)
ep  37/100 misclassified    40/50000 (train)   169/10000 (test)
ep  38/100 misclassified    35/50000 (train)   169/10000 (test)
ep  39/100 misclassified    32/50000 (train)   161/10000 (test)
ep  40/100 misclassified    21/50000 (train)   163/10000 (test)
ep  41/100 misclassified    19/50000 (train)   164/10000 (test)
ep  42/100 misclassified    19/50000 (train)   162/10000 (test)
ep  43/100 misclassified    15/50000 (train)   163/10000 (test)
ep  44/100 misclassified    10/50000 (train)   159/10000 (test)
ep  45/100 misclassified     8/50000 (train)   160/10000 (test)
ep  46/100 misclassified     4/50000 (train)   162/10000 (test)
ep  47/100 misclassified     4/50000 (train)   157/10000 (test)
ep  48/100 misclassified     4/50000 (train)   160/10000 (test)
ep  49/100 misclassified     4/50000 (train)   160/10000 (test)
ep  50/100 misclassified     2/50000 (train)   157/10000 (test)
ep  51/100 misclassified     3/50000 (train)   162/10000 (test)
ep  52/100 misclassified     2/50000 (train)   158/10000 (test)
ep  53/100 misclassified     1/50000 (train)   153/10000 (test)
ep  54/100 misclassified     1/50000 (train)   155/10000 (test)
ep  55/100 misclassified     0/50000 (train)   149/10000 (test)
ep  56/100 misclassified     0/50000 (train)   156/10000 (test)
ep  57/100 misclassified     0/50000 (train)   155/10000 (test)
ep  58/100 misclassified     0/50000 (train)   152/10000 (test)
ep  59/100 misclassified     0/50000 (train)   151/10000 (test)
ep  60/100 misclassified     0/50000 (train)   151/10000 (test)
ep  61/100 misclassified     0/50000 (train)   147/10000 (test)
ep  62/100 misclassified     1/50000 (train)   150/10000 (test)
ep  63/100 misclassified     0/50000 (train)   147/10000 (test)
ep  64/100 misclassified     0/50000 (train)   152/10000 (test)
ep  65/100 misclassified     0/50000 (train)   148/10000 (test)
ep  66/100 misclassified     0/50000 (train)   145/10000 (test)
ep  67/100 misclassified     0/50000 (train)   147/10000 (test)
ep  68/100 misclassified     0/50000 (train)   149/10000 (test)
ep  69/100 misclassified     0/50000 (train)   147/10000 (test)
ep  70/100 misclassified     0/50000 (train)   144/10000 (test)
ep  71/100 misclassified     0/50000 (train)   144/10000 (test)
ep  72/100 misclassified     0/50000 (train)   143/10000 (test)
ep  73/100 misclassified     0/50000 (train)   143/10000 (test)
ep  74/100 misclassified     0/50000 (train)   145/10000 (test)
ep  75/100 misclassified     0/50000 (train)   142/10000 (test)
ep  76/100 misclassified     0/50000 (train)   144/10000 (test)
ep  77/100 misclassified     0/50000 (train)   144/10000 (test)
ep  78/100 misclassified     0/50000 (train)   147/10000 (test)
ep  79/100 misclassified     0/50000 (train)   144/10000 (test)
ep  80/100 misclassified     0/50000 (train)   146/10000 (test)
ep  81/100 misclassified     0/50000 (train)   149/10000 (test)
ep  82/100 misclassified     0/50000 (train)   148/10000 (test)
ep  83/100 misclassified     0/50000 (train)   146/10000 (test)
ep  84/100 misclassified     0/50000 (train)   145/10000 (test)
ep  85/100 misclassified     0/50000 (train)   146/10000 (test)
ep  86/100 misclassified     0/50000 (train)   147/10000 (test)
ep  87/100 misclassified     0/50000 (train)   147/10000 (test)
ep  88/100 misclassified     0/50000 (train)   145/10000 (test)
ep  89/100 misclassified     0/50000 (train)   145/10000 (test)
ep  90/100 misclassified     0/50000 (train)   148/10000 (test)
ep  91/100 misclassified     0/50000 (train)   146/10000 (test)
ep  92/100 misclassified     0/50000 (train)   143/10000 (test)
ep  93/100 misclassified     0/50000 (train)   144/10000 (test)
ep  94/100 misclassified     0/50000 (train)   147/10000 (test)
ep  95/100 misclassified     0/50000 (train)   146/10000 (test)
ep  96/100 misclassified     0/50000 (train)   146/10000 (test)
ep  97/100 misclassified     0/50000 (train)   146/10000 (test)
ep  98/100 misclassified     0/50000 (train)   147/10000 (test)
ep  99/100 misclassified     0/50000 (train)   145/10000 (test)
ep 100/100 misclassified     0/50000 (train)   145/10000 (test)
epoch 101/100 batch  50/50
real	57m52.807s
user	47m2.694s
sys	9m33.379s
```
