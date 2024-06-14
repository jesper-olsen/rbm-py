import numpy as np
import struct
import matplotlib.pyplot as plt
import scipy.io

def read_labels(file_path):
    with open(file_path, 'rb') as file:
        magic, num_items = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError('Invalid magic number in label file: {}'.format(magic))
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    return labels

def read_images(file_path):
    with open(file_path, 'rb') as file:
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError('Invalid magic number in image file: {}'.format(magic))
        #images = np.frombuffer(file.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(num_images, -1)
        images = np.array(images, copy=True)
    return images

def label2target(labels):
    numlab=10
    return np.eye(numlab)[labels] 

def make_batches(path):
    #use Hinton's pre-processed matlab archive - note sequences have been re-ordered relative to MNIST
    #return scipy.io.loadmat(path+"/mnistdata.mat")

    # Return training, validation and test sets.
    # Each set is shaped into batches of 100 images
    images=read_images(path+"/raw/train-images-idx3-ubyte")/255.0
    batchdata = images[0:50000]
    validbatchdata = images[50000:]
    finaltestbatchdata=read_images(path+"/raw/t10k-images-idx3-ubyte")/255.0
    
    labels=read_labels(path+"/raw/train-labels-idx1-ubyte") 
    batchtargets = label2target(labels[0:50000])
    validbatchtargets = label2target(labels[50000:])
    finaltestbatchtargets=label2target( read_labels(path+"/raw/t10k-labels-idx1-ubyte") )

    def reshape(a, n):
        a = a.reshape(-1, 100, n)
        a = a.transpose(1,2,0)
        return a

    npix=28*28
    numlab=10
    return { "batchdata": reshape(batchdata, npix),
             "validbatchdata": reshape(validbatchdata,npix),
             "testbatchdata": reshape(finaltestbatchdata,npix),
             "batchtargets": reshape(batchtargets, numlab),
             "validbatchtargets": reshape(validbatchtargets, numlab),
             "testbatchtargets": reshape(finaltestbatchtargets,numlab)}

def show_image(image, label, index):
    plt.figure()
    plt.imshow(image.reshape(28,28), cmap='gray')
    plt.title(f'Label: {label} index {index}')
    plt.show()
 
def test_raw():
    image_file_path = 'MNIST/raw/train-images-idx3-ubyte'
    label_file_path = 'MNIST/raw/train-labels-idx1-ubyte'
    image_file_path = 'MNIST/raw/t10k-images-idx3-ubyte'
    label_file_path = 'MNIST/raw/t10k-labels-idx1-ubyte'
    labels = read_labels(label_file_path)
    images = read_images(image_file_path)

    print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')
    print(f'Loaded {labels.shape[0]} labels')

    for i in range(5):
        show_image(images[i], labels[i], i)

def test_batches():
    d=make_batches("MNIST")
    # show 1st 5 images in batch 0
    for i in range(5):
        label = np.argmax(d["batchtargets"][i,:,0], axis=0)
        show_image(d["batchdata"][i,:,0], label, i)

if __name__=="__main__":
    #test_raw()
    test_batches()
