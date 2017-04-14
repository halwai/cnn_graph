import numpy as np
import scipy as sp
from scipy.io import *
from scipy.sparse import csr_matrix
import sklearn
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.metrics.pairwise import euclidean_distances
import os.path 
import sys

def preprocess(k=100):

    base_ftrs_dir = '/data4/abhijeet/Datasets/PASCAL_VOC/Ftrs/vgg16/'
    base_labels_dir = '/data4/abhijeet/Datasets/PASCAL_VOC/Labels/'
    base_storage_dir = '/data4/abhijeet/Datasets/PASCAL_VOC/GCN/preprocessing/experiment3/'


    temp_filename = base_storage_dir + 'scikit_fc7_k_' + str(k) + '.mat'
    if os.path.exists(temp_filename):
        print('File already exists for K = ',k)
        return()

    # Expected Outputs
    test_labels = sp.io.loadmat( base_labels_dir + 'test_labels.mat')
    train_labels = sp.io.loadmat( base_labels_dir + 'train_labels.mat')

    Y_test = test_labels['test_labels']
    Y_train = train_labels['train_labels']

    print(Y_test.shape)
    print(Y_train.shape)



    test_ftrs = sp.io.loadmat( base_ftrs_dir + 'test_ftrs.mat')
    train_ftrs = sp.io.loadmat( base_ftrs_dir + 'train_ftrs.mat')

    test_ftrs = test_ftrs['test_ftrs']
    train_ftrs = train_ftrs['train_ftrs']

    print(test_ftrs.shape)
    print(train_ftrs.shape)



    #Kd-tree NNSearch
    t = time.time()
    nbrs = NearestNeighbors(n_neighbors = k+1,algorithm='kd_tree').fit(train_ftrs)
    print(time.time() - t)
    t = time.time()

    #NN for train-images
    X_distances, indices_train = nbrs.kneighbors(train_ftrs)
    #remove closest neighbor that is the node itself
    indices_train = indices_train[:,1:]
    print(time.time() - t)
    t = time.time()

    #NN for test-images
    X_distances, indices_test = nbrs.kneighbors(test_ftrs)
    #remove last node to maintain consistent number of neighbors
    indices_test = indices_test[:,:-1]
    print(time.time() - t)

    # Input for each data-point
    X_train = np.zeros((train_ftrs.shape[0], k, Y_train.shape[1]))
    for i in range(X_train.shape[0]):
        X_train[i,:,:] = Y_train[indices_train[i,:], :]

    X_test = np.zeros((test_ftrs.shape[0], k, Y_test.shape[1]))
    for i in range(X_test.shape[0]):
        X_test[i,:,:] = Y_train[indices_test[i,:], :]

    print(X_train.shape)
    print(X_test.shape)


    # Graphs per data point
    #Construct a graph for each data point
    t = time.time()
    Adjacency_train = []#np.zeros((X_train.shape[0], k, k))
    for i in range(X_train.shape[0]):
        temp_x = train_ftrs[indices_train[i,:], :]
        temp_adjacency = euclidean_distances(temp_x,temp_x)
        gamma = np.max(np.max(temp_adjacency))
        temp_adjacency = temp_adjacency * temp_adjacency
        temp_adjacency = np.exp(-temp_adjacency/(gamma**2))
        #Adjacency_train[i,:,:] = temp_adjacency
        #temp_adjacency = (temp_adjacency + temp_adjacency.T) / 2
        Adjacency_train.append(csr_matrix(temp_adjacency))        

    print(time.time() - t)
    t = time.time()

    Adjacency_test = []#np.zeros((X_test.shape[0], k, k))
    for i in range(X_test.shape[0]):
        temp_x = train_ftrs[indices_test[i,:], :]
        temp_adjacency = euclidean_distances(temp_x,temp_x)
        gamma = np.max(np.max(temp_adjacency))
        temp_adjacency = temp_adjacency * temp_adjacency
        temp_adjacency = np.exp(-temp_adjacency/(gamma**2))
        # TODO have a look later
        #temp_adjacency = (temp_adjacency + temp_adjacency.T) / 2
        #Adjacency_test[i,:,:] = temp_adjacency
        Adjacency_test.append(csr_matrix(temp_adjacency))
        
    print(time.time() - t)


    # store the data in a file for further use
    temp_filename = base_storage_dir + 'scikit_fc7_k_' + str(k)
    t = time.time()
    sp.io.savemat( temp_filename, {'Y_train':Y_train,'Y_test':Y_test, 'X_train':X_train, 'X_test':X_test, 'Adjacency_test':Adjacency_test, 'Adjacency_train':Adjacency_train })
    print(time.time() - t)

if __name__ == "__main__":
    for k in range(110, 140, 30):
        preprocess(k)
        print("\n\n Preprocessing Done for K= ",k,"\n\n")

