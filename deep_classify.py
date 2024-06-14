import numpy as np
from scipy.special import expit as logistic # 1/(1+exp(-x))
from scipy.optimize import minimize
import mnist

def train_rbm(batchdata, maxepoch, numhid, binary=True):
    if binary:
        epsilonw, epsilonvb, epsilonhb = 0.1, 0.1, 0.1
    else:
        epsilonw, epsilonvb, epsilonhb = 0.001, 0.001, 0.001

    weightcost = 0.0002
    initialmomentum, finalmomentum = 0.5, 0.9

    numcases, idim, numbatches = batchdata.shape

    # Initializing symmetric weights and biases
    vishid = 0.1 * np.random.randn(idim, numhid)
    hidbiases = np.zeros((1, numhid))
    visbiases = np.zeros((1, idim))

    poshidprobs = np.zeros((numcases, numhid))
    neghidprobs = np.zeros((numcases, numhid))
    posprods = np.zeros((idim, numhid))
    negprods = np.zeros((idim, numhid))
    vishidinc = np.zeros((idim, numhid))
    hidbiasinc = np.zeros((1, numhid))
    visbiasinc = np.zeros((1, idim))
    batchposhidprobs = np.zeros((numcases, numhid, numbatches))

    for epoch in range(maxepoch):
        nerrors = 0
        for batch in range(numbatches):

            # Start positive phase
            data = batchdata[:, :, batch]
            poshidprobs = data @ vishid + hidbiases
            if binary:
                poshidprobs = logistic(poshidprobs)
            batchposhidprobs[:, :, batch] = poshidprobs
            #posprods = data.T @ poshidprobs
            np.dot(data.T, poshidprobs, out=posprods) #in place
            poshidact = np.sum(poshidprobs, axis=0)
            posvisact = np.sum(data, axis=0)
            # End positive phase

            if binary:
                poshidstates = poshidprobs > np.random.rand(numcases, numhid)
            else:
                poshidstates = poshidprobs + np.random.randn(numcases, numhid)

            # Start negative phase
            negdata = logistic(poshidstates @ vishid.T+visbiases)
            neghidprobs = negdata @ vishid + hidbiases
            if binary:
                neghidprobs = logistic(neghidprobs)

            negprods = negdata.T @ neghidprobs
            neghidact = np.sum(neghidprobs, axis=0)
            negvisact = np.sum(negdata, axis=0)
            # End negative phase

            nerrors += np.sum((data - negdata) ** 2)
            momentum = finalmomentum if epoch >= 5 else initialmomentum

            # Update weights and biases
            vishid += momentum*vishidinc + epsilonw*((posprods-negprods)/numcases-weightcost*vishid)
            visbiases += momentum*visbiasinc + (epsilonvb/numcases) * (posvisact-negvisact)
            hidbiases += momentum*hidbiasinc + (epsilonhb/numcases) * (poshidact-neghidact)
        print(f'ep {epoch+1:2}/{maxepoch} error {nerrors:.1f}')
    return (batchposhidprobs, vishid, hidbiases, visbiases)

def cg_toplevel(VV, l, w3probs, target):
    N = w3probs.shape[0]

    # Deconversion.
    w_class = VV.reshape((l[0] + 1, l[1]))
    w3probs = np.hstack([w3probs, np.ones((N, 1))])

    targetout = np.exp(w3probs @ w_class)
    targetout = targetout / np.sum(targetout, axis=1, keepdims=True)
    
    f = -np.sum(target * np.log(targetout))
    Ix = targetout - target
    dw_class = w3probs.T @ Ix
    df = dw_class.ravel()
    return f, df

def cg(VV, l, XX, target):
    N = XX.shape[0]

    # Deconversion
    w = [None]*4
    z=0
    for i in range(4):
        w[i] = VV[z:z + (l[i] + 1) * l[i+1]].reshape((l[i] + 1, l[i+1]))
        z += (l[i] + 1) * l[i+1]

    XX = np.hstack([XX, np.ones((N, 1))])
    wprobsIM1 = XX
    wprobs = [None]*3
    for i in range(3):
        wprobs[i] = np.hstack([logistic(wprobsIM1 @ w[i]), np.ones((N, 1))])
        wprobsIM1 = wprobs[i]

    targetout = np.exp(wprobs[2] @ w[3])
    targetout = targetout / np.sum(targetout, axis=1, keepdims=True)
    f = -np.sum(target * np.log(targetout))

    Ix = targetout - target
    dw = [None] * 4
    dw[3] = wprobs[2].T @ Ix

    Ix = (Ix @ w[3].T) * wprobs[2] * (1 - wprobs[2])
    Ix = Ix[:,:-1]     # rm last column of ones 
    dw[2] = wprobs[1].T @ Ix   

    Ix = (Ix @ w[2].T) * wprobs[1] * (1 - wprobs[1])
    Ix = Ix[:,:-1]
    dw[1] = wprobs[0].T @ Ix

    Ix = (Ix @ w[1].T) * wprobs[0] * (1 - wprobs[0])
    Ix = Ix[:,:-1]
    dw[0] = XX.T @ Ix

    df = np.concatenate([dw[i].ravel() for i in range(4)])
    return f, df

def calc_error(batchdata, batchtargets, w):
    N, numdims, numbatches = batchdata.shape
    wprobs = [None]*(len(w)-1)
    err, err_cr, ncorrect = 0, 0, 0
    for batch in range(numbatches):
        data = batchdata[:, :, batch]
        target = batchtargets[:, :, batch]
        wprobsIM1= np.concatenate((data, np.ones((N, 1))), axis=1)
        for i in range(len(w)-1):
            wprobs[i] = logistic(wprobsIM1 @ w[i])
            wprobs[i] = np.concatenate((wprobs[i], np.ones((N, 1))), axis=1)
            wprobsIM1=wprobs[i]
        targetout = np.exp(wprobs[-1] @ w[-1])
        targetout = targetout / np.sum(targetout, axis=1, keepdims=True)

        J = np.argmax(targetout, axis=1)
        J1 = np.argmax(target, axis=1)
        ncorrect += np.sum(J == J1)
        err_cr -= np.sum(target * np.log(targetout))

    err = N * numbatches - ncorrect
    #crerr = err_cr / numbatches
    return err, N*numbatches

def backprop_classify(mnist_data, w):
    maxepoch = 100
    print('\nTraining discriminative model on MNIST by minimizing cross entropy error.')

    l = [x.shape[0]-1 for x in w] + [w[-1].shape[1]]

    for epoch in range(1, maxepoch + 1):
        # Compute training misclassification error
        train_err, train_n = calc_error(mnist_data['batchdata'], mnist_data['batchtargets'], w)
        test_err, test_n = calc_error(mnist_data['testbatchdata'], mnist_data['testbatchtargets'], w)
        print(f"ep {epoch:3}/{maxepoch} misclassified {train_err:5}/{train_n} (train) {test_err:5}/{test_n} (test)")

        # Combine 10 minibatches into 1 larger minibatch
        batchdata, batchtargets=mnist_data['batchdata'], mnist_data['batchtargets']
        N, numdims, numbatches = batchdata.shape
        tt = 0
        for batch in range(numbatches//10): # 50 batches of 1000 cases each.
            print(f'epoch {epoch:3} batch {batch:3}\r', end='')

            data, targets = [], []
            for kk in range(10):
                data.append(batchdata[:, :, tt * 10 + kk])
                targets.append(batchtargets[:, :, tt * 10 + kk])
            data, targets = np.vstack(data), np.vstack(targets)
            tt += 1

            max_iter=3
            if epoch < 6:  # First update top-level weights holding other weights fixed.
                N = data.shape[0]
                
                wprobs = data
                for i in range(3):
                    wprobs = np.hstack([wprobs, np.ones((N, 1))])
                    wprobs = logistic(wprobs @ w[i])
    
                VV = w[-1].ravel()
                Dim = [wprobs.shape[1], w[-1].shape[1]]

                res = minimize(cg_toplevel, VV, args=(Dim, wprobs, targets), method='CG', jac=True, options={'maxiter': max_iter})
                w[-1]= res.x.reshape((Dim[0] + 1, Dim[1]))
            else:
                VV = np.concatenate([layer.ravel() for layer in w])
                Dim = [layer.shape[0] - 1 for layer in w] + [w[-1].shape[1]]
                res = minimize(cg, VV, args=(Dim, data, targets), method='CG', jac=True, options={'maxiter': max_iter})
                layers = [(layer.shape[0] - 1, layer.shape[1]) for layer in w]
                w=[]
                start=0
                for (l1,l2) in layers:
                    end = start + (l1+1)*l2
                    w+=[res.x[start:end].reshape((l1+1, l2))]
                    start=end

def deep_classify():
    np.random.seed(17)
    maxepoch=10
    mnist_data=mnist.make_batches("MNIST")
    batch_size, idim, numbatches = mnist_data["batchdata"].shape
    print(f"Batchsize: {batch_size} Input-dim: {idim} #training batches: {numbatches}")

    LAYER = [idim, 500, 500, 2000]
    l=[]
    batchdata=mnist_data["batchdata"]
    for i in range(0,len(LAYER)-1):
        print(f"Pretraining Layer {i+1} with RBM: {LAYER[i]}-{LAYER[i+1]}")
        batchdata, vishid, hidrecbiases, visbiases  = train_rbm(batchdata,  maxepoch, LAYER[i+1], binary=i<len(LAYER)-2)
        l+=[(vishid, hidrecbiases, visbiases)]

    w=[np.vstack((vh,hb)) for (vh,hb,_) in l] 
    w+=[0.1 * np.random.randn(w[-1].shape[1] + 1, 10)]
    backprop_classify(mnist_data, w) 

if __name__=="__main__":
    deep_classify()

