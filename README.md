Deep Boltzmann Machines
=======================

Deep Boltzmann Machines [1,2] evaluated on the MNIST [3] handwritten digit recognition task; Python+Numpy implementation.

Network: 3 stacked Restricted Boltzmann Machines (RBMs): 784-500, 500-500, 500-2000 followed by 10
softmax outputs for the digits 0-9.

Train on 50k samples, test on 10k; Error rate: 0.0% (train) / 1.52% (test)

See [FF repo](https://github.com/jesper-olsen/ff-py) for another result on the same task.

References:
-----------
[1] [Deep Boltzmann Machines](https://proceedings.mlr.press/v5/salakhutdinov09a.html)
[2] [Matlab code](https://www.cs.toronto.edu/~hinton/MatlabForSciencePaper.html)
[3] [LeCun's raw MNIST db](http://yann.lecun.com/exdb/mnist/)

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
% time python deep_classify.py  | tee OUT
Batchsize: 100 Input-dim: 784 #training batches: 500
Pretraining Layer 1 with RBM: 784-500
ep  1/10 error 893107.0
ep  2/10 error 550579.0
ep  3/10 error 476174.6
ep  4/10 error 439856.7
ep  5/10 error 415889.1
ep  6/10 error 401354.5
ep  7/10 error 390931.3
ep  8/10 error 382890.3
ep  9/10 error 377172.2
ep 10/10 error 371367.1
Pretraining Layer 2 with RBM: 500-500
ep  1/10 error 1090926.1
ep  2/10 error 679122.3
ep  3/10 error 606953.3
ep  4/10 error 565428.8
ep  5/10 error 535453.4
ep  6/10 error 511880.9
ep  7/10 error 491370.6
ep  8/10 error 473772.7
ep  9/10 error 458256.4
ep 10/10 error 445417.9
Pretraining Layer 3 with RBM: 500-2000
ep  1/10 error 1954303.2
ep  2/10 error 1433173.4
ep  3/10 error 1284023.2
ep  4/10 error 1205109.6
ep  5/10 error 1152789.0
ep  6/10 error 1115782.9
ep  7/10 error 1086539.7
ep  8/10 error 1061836.3
ep  9/10 error 1039961.7
ep 10/10 error 1021190.8

Training discriminative model on MNIST by minimizing cross entropy error.
ep   1/100 misclassified 44949/50000 (train)  8994/10000 (test)
ep   2/100 misclassified  5677/50000 (train)  1039/10000 (test)
ep   3/100 misclassified  4926/50000 (train)   920/10000 (test)
ep   4/100 misclassified  4588/50000 (train)   838/10000 (test)
ep   5/100 misclassified  4327/50000 (train)   783/10000 (test)
ep   6/100 misclassified  4099/50000 (train)   747/10000 (test)
ep   7/100 misclassified  3446/50000 (train)   632/10000 (test)
ep   8/100 misclassified  2938/50000 (train)   540/10000 (test)
ep   9/100 misclassified  2608/50000 (train)   495/10000 (test)
ep  10/100 misclassified  2409/50000 (train)   469/10000 (test)
ep  11/100 misclassified  2166/50000 (train)   435/10000 (test)
ep  12/100 misclassified  1904/50000 (train)   384/10000 (test)
ep  13/100 misclassified  1714/50000 (train)   365/10000 (test)
ep  14/100 misclassified  1537/50000 (train)   343/10000 (test)
ep  15/100 misclassified  1387/50000 (train)   325/10000 (test)
ep  16/100 misclassified  1209/50000 (train)   303/10000 (test)
ep  17/100 misclassified  1111/50000 (train)   290/10000 (test)
ep  18/100 misclassified  1035/50000 (train)   284/10000 (test)
ep  19/100 misclassified   918/50000 (train)   279/10000 (test)
ep  20/100 misclassified   847/50000 (train)   254/10000 (test)
ep  21/100 misclassified   734/50000 (train)   237/10000 (test)
ep  22/100 misclassified   704/50000 (train)   236/10000 (test)
ep  23/100 misclassified   595/50000 (train)   219/10000 (test)
ep  24/100 misclassified   502/50000 (train)   215/10000 (test)
ep  25/100 misclassified   465/50000 (train)   207/10000 (test)
ep  26/100 misclassified   402/50000 (train)   204/10000 (test)
ep  27/100 misclassified   348/50000 (train)   207/10000 (test)
ep  28/100 misclassified   322/50000 (train)   196/10000 (test)
ep  29/100 misclassified   262/50000 (train)   187/10000 (test)
ep  30/100 misclassified   254/50000 (train)   194/10000 (test)
ep  31/100 misclassified   220/50000 (train)   195/10000 (test)
ep  32/100 misclassified   189/50000 (train)   186/10000 (test)
ep  33/100 misclassified   132/50000 (train)   174/10000 (test)
ep  34/100 misclassified   120/50000 (train)   178/10000 (test)
ep  35/100 misclassified   146/50000 (train)   178/10000 (test)
ep  36/100 misclassified    82/50000 (train)   172/10000 (test)
ep  37/100 misclassified    68/50000 (train)   170/10000 (test)
ep  38/100 misclassified    61/50000 (train)   165/10000 (test)
ep  39/100 misclassified    45/50000 (train)   166/10000 (test)
ep  40/100 misclassified    40/50000 (train)   165/10000 (test)
ep  41/100 misclassified    33/50000 (train)   168/10000 (test)
ep  42/100 misclassified    25/50000 (train)   162/10000 (test)
ep  43/100 misclassified    18/50000 (train)   161/10000 (test)
ep  44/100 misclassified    17/50000 (train)   162/10000 (test)
ep  45/100 misclassified    16/50000 (train)   160/10000 (test)
ep  46/100 misclassified    14/50000 (train)   164/10000 (test)
ep  47/100 misclassified    12/50000 (train)   158/10000 (test)
ep  48/100 misclassified    37/50000 (train)   173/10000 (test)
ep  49/100 misclassified    11/50000 (train)   163/10000 (test)
ep  50/100 misclassified    10/50000 (train)   165/10000 (test)
ep  51/100 misclassified    11/50000 (train)   163/10000 (test)
ep  52/100 misclassified    10/50000 (train)   168/10000 (test)
ep  53/100 misclassified     9/50000 (train)   165/10000 (test)
ep  54/100 misclassified     9/50000 (train)   157/10000 (test)
ep  55/100 misclassified     8/50000 (train)   162/10000 (test)
ep  56/100 misclassified     7/50000 (train)   165/10000 (test)
ep  57/100 misclassified     5/50000 (train)   163/10000 (test)
ep  58/100 misclassified     6/50000 (train)   166/10000 (test)
ep  59/100 misclassified     5/50000 (train)   166/10000 (test)
ep  60/100 misclassified     3/50000 (train)   159/10000 (test)
ep  61/100 misclassified     9/50000 (train)   162/10000 (test)
ep  62/100 misclassified     3/50000 (train)   166/10000 (test)
ep  63/100 misclassified     3/50000 (train)   164/10000 (test)
ep  64/100 misclassified     3/50000 (train)   171/10000 (test)
ep  65/100 misclassified     3/50000 (train)   170/10000 (test)
ep  66/100 misclassified     2/50000 (train)   167/10000 (test)
ep  67/100 misclassified     1/50000 (train)   167/10000 (test)
ep  68/100 misclassified     2/50000 (train)   169/10000 (test)
ep  69/100 misclassified     2/50000 (train)   168/10000 (test)
ep  70/100 misclassified     1/50000 (train)   164/10000 (test)
ep  71/100 misclassified     2/50000 (train)   163/10000 (test)
ep  72/100 misclassified     1/50000 (train)   165/10000 (test)
ep  73/100 misclassified     1/50000 (train)   163/10000 (test)
ep  74/100 misclassified     1/50000 (train)   164/10000 (test)
ep  75/100 misclassified     1/50000 (train)   160/10000 (test)
ep  76/100 misclassified     1/50000 (train)   168/10000 (test)
ep  77/100 misclassified     2/50000 (train)   167/10000 (test)
ep  78/100 misclassified     2/50000 (train)   165/10000 (test)
ep  79/100 misclassified     1/50000 (train)   158/10000 (test)
ep  80/100 misclassified     1/50000 (train)   162/10000 (test)
ep  81/100 misclassified     1/50000 (train)   159/10000 (test)
ep  82/100 misclassified     1/50000 (train)   160/10000 (test)
ep  83/100 misclassified     0/50000 (train)   169/10000 (test)
ep  84/100 misclassified     2/50000 (train)   171/10000 (test)
ep  85/100 misclassified     1/50000 (train)   166/10000 (test)
ep  86/100 misclassified     1/50000 (train)   164/10000 (test)
ep  87/100 misclassified     0/50000 (train)   157/10000 (test)
ep  88/100 misclassified     0/50000 (train)   160/10000 (test)
ep  89/100 misclassified     0/50000 (train)   162/10000 (test)
ep  90/100 misclassified     0/50000 (train)   154/10000 (test)
ep  91/100 misclassified     0/50000 (train)   156/10000 (test)
ep  92/100 misclassified     0/50000 (train)   153/10000 (test)
ep  93/100 misclassified     0/50000 (train)   154/10000 (test)
ep  94/100 misclassified     0/50000 (train)   158/10000 (test)
ep  95/100 misclassified     0/50000 (train)   154/10000 (test)
ep  96/100 misclassified     0/50000 (train)   158/10000 (test)
ep  97/100 misclassified     0/50000 (train)   152/10000 (test)
ep  98/100 misclassified     0/50000 (train)   149/10000 (test)
ep  99/100 misclassified     0/50000 (train)   163/10000 (test)
ep 100/100 misclassified     0/50000 (train)   152/10000 (test)
epoch 100 batch  49
real	76m52.511s
user	64m59.783s
sys	9m52.922s
```
