from . import graph

import tensorflow as tf
import sklearn
import scipy.sparse
import numpy as np
import os, time, collections, shutil


#NFEATURES = 28**2
#NCLASSES = 10


# Common methods for all models


class base_model(object):
    
    def __init__(self , num_labels_per_image):
        self.regularizers = []
        self.num_labels_per_image = num_labels_per_image
    
    # High-level interface which runs the constructed computational graph.
    def evaluation_results(self, predictions, labels, label_probability):
        N = predictions.shape[0]
        if len(labels.shape) is 2:
            F_last = labels.shape[1]
        else:
            F_last = 1
            labels = labels.reshape((labels.shape[0],1))

        PR = np.zeros(shape=(F_last,2))

        for i in range(F_last):
            m1 = np.sum(labels[:,i])
            m2 = np.sum(predictions[:,i])
            m3 = np.sum(np.multiply(labels[:,i], predictions[:,i]))

            if m1>0 and m2>0 and m3>0:
                PR[i,0] = float(m3)/float(m2)
                PR[i,1] = float(m3)/float(m1)
        mPR = np.mean(PR, axis=0)
        precision = mPR[0]
        recall = mPR[1]
        f_measure = (2*precision*recall)/(precision+recall)

        mAP = np.zeros(shape=(F_last,1))
        print(np.sum(predictions * labels), '/' , np.sum(labels), '=' , np.sum(predictions*labels) / np.sum(labels) )

        for i in range(F_last):
            ground_truth = labels[:,i]
            prob = label_probability[:,i]
            pred = predictions[:,i]

            ind = np.argsort(prob)[::-1]    #Descending sort
            TP = 0
            count = 0
            sum_so_far = 0.0

            for j in range(len(prob)):
                idx = ind[j]
                if ground_truth[idx] and pred[idx]:
                    TP += 1
                if ground_truth[idx]:
                    count+=1
                    sum_so_far += TP/count
            
            if np.sum(ground_truth) is 0:
                mAP[i] = 0
            else:
                mAP[i] = sum_so_far/np.sum(ground_truth)

        MAP = np.zeros(shape=(N,1))

        for i in range(N):
            ground_truth = labels[i,:]
            prob = label_probability[i,:]
            pred = predictions[i,:]
            ind = np.argsort(prob)[::-1]    #Descending sort
            TP = 0
            count = 0
            sum_so_far = 0.0

            for j in range(len(prob)):
                idx = ind[j]
                if ground_truth[idx] and pred[idx]:
                    TP += 1
                if ground_truth[idx]:
                    count+=1
                    sum_so_far += TP/count

            if np.sum(ground_truth) is 0:
                MAP[i] = 0
            else:
                MAP[i] = sum_so_far/np.sum(ground_truth)

        return precision, recall, f_measure, mAP, MAP

    def predict(self, data, laplacians, labels=None, sess=None):
        loss = 0
        size = data.shape[0]
        predictions = np.empty((size, labels.shape[1]))
        sess = self._get_session(sess)
        if len(data.shape) is 2:
            data = np.expand_dims(data, 2) # N * M * F(=1)
        if len(labels.shape) is 1:
            labels = np.expand_dims(labels, 1) # N * F(=1)
            
        for begin in range(0, size, self.batch_size):
            end = begin + self.batch_size
            end = min([end, size])
            #print(begin,end)
            if len(laplacians) > 0:
                #single_laplacian =  [q[1] * scipy.sparse.csr_matrix(q[1].shape) for q in laplacians]
                single_laplacian =  [q[1] * np.zeros(q[1].shape) for q in laplacians]
                #print(single_laplacian)
                batch_laplacians = [[] for i in range(len(single_laplacian))]
                for j in range(len(single_laplacian)):
                    batch_laplacians[j].extend(laplacians[j][begin:end])
                    if end - begin < self.batch_size:
                        for k in range(self.batch_size-end+begin):
                            batch_laplacians[j].append(single_laplacian[j])
            else:
                batch_laplacians = []
            #print(batch_laplacians)
            #print(type(batch_laplacians),len(batch_laplacians))
            #for i in range(len(batch_laplacians)):
            #    print(type(batch_laplacians[i]),len(batch_laplacians[i]))
                
            batch_data = np.zeros((self.batch_size, data.shape[1], data.shape[2]))
            tmp_data = data[begin:end,:,:]
            if type(tmp_data) is not np.ndarray:
                tmp_data = tmp_data.toarray()  # convert sparse matrices
            batch_data[:end-begin,:,:] = tmp_data
            
            feed_dict = {i:d for i,d in zip(self.ph_laplacians, batch_laplacians)}
            feed_dict.update({self.ph_data: batch_data, self.ph_dropout: 1}) 
            
            # Compute loss if labels are given.
            if labels is not None:
                batch_labels = np.zeros((self.batch_size, labels.shape[1]))
                batch_labels[:end-begin,:] = labels[begin:end,:]
                feed_dict.update({self.ph_labels:batch_labels})
                batch_pred, batch_loss = sess.run([self.op_prediction, self.op_loss], feed_dict)
                loss += batch_loss
            else:
                batch_pred = sess.run(self.op_prediction, feed_dict)
            
            predictions[begin:end,:] = batch_pred[:end-begin,:]
            
        if labels is not None:
            return predictions, loss * self.batch_size / size
        else:
            return predictions
        
    def evaluate(self, data, laplacians, labels, sess=None):
        """
        Runs one evaluation against the full epoch of data.
        Return the precision and the number of correct predictions.
        Batch evaluation saves memory and enables this to run on smaller GPUs.

        sess: the session in which the model has been trained.
        op: the Tensor that returns the number of correct predictions.
        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        labels: size N x F_last
            N: number of signals (samples)
            F_last: Multilabel Problem (= #of classes)
        """
        t_process, t_wall = time.process_time(), time.time()
        label_probability, loss = self.predict(data, laplacians, labels, sess)
        #evaluation metrics -- f1, precision, recall...
        
        #ind = label_probability.argsort()[:,-self.num_labels_per_image:][::-1] #get largest indices per image
        predictions = np.zeros(label_probability.shape)

        for i in range(len(label_probability)):
            ind = label_probability[i].argsort()[-self.num_labels_per_image:][::-1]
            predictions[i, [ind]]=1

        precision, recall, f_measure, mAP, MAP = self.evaluation_results(predictions, labels, label_probability)

        string = 'precision: {:.2f}, recall : {:.2f}, f_measure: {:.2f}, mAP {:.2f}, MAP {:.2f}'.format(
                precision, recall, f_measure, np.mean(mAP), np.mean(MAP))
        if sess is None:
            string += '\ntime: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall)
        return string, precision, recall, f_measure, mAP, MAP

    def fit(self, train_data, train_labels, val_data, val_labels):
        t_process, t_wall = time.process_time(), time.time()
        # tesnsorflow usage settings
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.400)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        sess = tf.Session(graph=self.graph)
        shutil.rmtree(self._get_path('summaries'), ignore_errors=True)
        writer = tf.summary.FileWriter(self._get_path('summaries'), self.graph)
        shutil.rmtree(self._get_path('checkpoints'), ignore_errors=True)
        os.makedirs(self._get_path('checkpoints'))
        path = os.path.join(self._get_path('checkpoints'), 'model')
        sess.run(self.op_init)
        
        if len(train_data.shape) is 2:
            assert train_data.shape[-1] == val_data.shape[-1]
            train_data = np.expand_dims(train_data,2) # N * M * F(=1)
            val_data = np.expand_dims(val_data,2) # N * M * F(=1)
        
        if len(train_labels.shape) is 1:
            assert train_labels.shape[-1] == val_labels.shape[-1]
            train_labels = np.expand_dims(train_labels,1) # N * F(=1)
            val_labels = np.expand_dims(val_labels,1) # N * F(=1)
            
        # Training.
        precisions = []
        recalls=[]
        f_measures=[]
        mAPs=[]
        MAPs=[]
        indices = collections.deque()
        num_steps = int(self.num_epochs * train_data.shape[0] / self.batch_size)
        for step in range(1, num_steps+1):

            # Be sure to have used all the samples before using one a second time.
            if len(indices) < self.batch_size:
                indices.extend(np.random.permutation(train_data.shape[0]))
            idx = [indices.popleft() for i in range(self.batch_size)]

            batch_data, batch_labels = train_data[idx,:,:], train_labels[idx,:]
            
            if len(self.train_laplacians) is not 0:
                batch_laplacians = [[lap[i] for i in idx] for lap in self.train_laplacians  ]
            else:
                batch_laplacians = []

            #print(len(batch_laplacians),type(batch_laplacians))
            #for i in range(len(batch_laplacians)):
            #    print(len(batch_laplacians[i]),type(batch_laplacians[i]))
            
            if type(batch_data) is not np.ndarray:
                batch_data = batch_data.toarray()  # convert sparse matrices

            feed_dict = {i:d for i,d in zip(self.ph_laplacians, batch_laplacians)}
            feed_dict.update({self.ph_data: batch_data, self.ph_dropout: 1, self.ph_labels:batch_labels}) 

            learning_rate, loss_average = sess.run([self.op_train, self.op_loss_average], feed_dict)
            #print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))

            # Periodical evaluation of the model.
            if step % self.eval_frequency == 0 or step == num_steps:
                epoch = step * self.batch_size / train_data.shape[0]
                print('step {} / {} (epoch {:.2f} / {}):'.format(step, num_steps, epoch, self.num_epochs))
                print('  learning_rate = {:.2e}, loss_average = {:.2e}'.format(learning_rate, loss_average))
                string, precision, recall, f_measure, mAP, MAP = self.evaluate(val_data, self.val_laplacians, val_labels, sess)
                precisions.append(precision)
                recalls.append(recall)
                f_measures.append(f_measure)
                mAPs.append(mAP)
                MAPs.append(MAP)
                print('  validation {}'.format(string))
                print('  time: {:.0f}s (wall {:.0f}s)'.format(time.process_time()-t_process, time.time()-t_wall))

                # Summaries for TensorBoard.
                summary = tf.Summary()
                summary.ParseFromString(sess.run(self.op_summary, feed_dict))
                summary.value.add(tag='validation/precision', simple_value=precision)
                summary.value.add(tag='validation/recall', simple_value=recall)
                summary.value.add(tag='validation/f_measure', simple_value=f_measure)
                summary.value.add(tag='validation/mAP', simple_value=np.mean(mAP))
                summary.value.add(tag='validation/MAP', simple_value=np.mean(MAP))
                writer.add_summary(summary, step)
                
                # Save model parameters (for evaluation).
                self.op_saver.save(sess, path, global_step=step)

        print('validation peaks: precision = {:.2f}, recall = {:.2f}, f_measure = {}, mAP = {}, MAP = {}'.format(max(precisions), max(recalls), max(f_measures), max(np.mean(mAPs, axis=1)), max(np.mean(MAPs, axis=1)) ))
        #print('mAPs is \n',mAPs)
        writer.close()
        sess.close()
        
        t_step = (time.time() - t_wall) / num_steps
        print(type(self.train_laplacians) , type(self.test_laplacians))
        return np.mean(precisions), np.mean(recalls), np.mean(f_measures), np.mean(np.mean(mAPs, axis=1)), np.mean(np.mean(MAPs, axis=1)) , self.train_laplacians, self.test_laplacians

    def get_var(self, name):
        sess = self._get_session()
        var = self.graph.get_tensor_by_name(name + ':0')
        val = sess.run(var)
        sess.close()
        return val

    # Methods to construct the computational graph.
    
    def build_graph(self, M_0, F_0, F_last, L):
        """Build the computational graph of the model."""
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Inputs.
            with tf.name_scope('inputs'):
                self.ph_data = tf.placeholder(tf.float32, (self.batch_size, M_0, F_0), 'data')
                self.ph_labels = tf.placeholder(tf.int32, (self.batch_size, F_last), 'labels')
                self.ph_laplacians = []

                for i in range(np.shape(np.array(L))[0]):
                    string_name = 'lalacians_' + str(i)
                    self.ph_laplacians.append(tf.placeholder(tf.float32, (self.batch_size, L[i].shape[0], L[i].shape[1]), string_name))
                    print(i,'th conv layer',self.batch_size,L[i].shape[0], L[i].shape[1])
                self.ph_dropout = tf.placeholder(tf.float32, (), 'dropout')
                #print('type(self.ph_laplacians) is ',type(self.ph_laplacians))

            # Model.
            op_logits = self.inference(self.ph_data, self.ph_laplacians ,self.ph_dropout)
            self.op_loss, self.op_loss_average = self.loss(op_logits, self.ph_labels, self.regularization)
            self.op_train = self.training(self.op_loss, self.learning_rate,
                    self.decay_steps, self.decay_rate, self.momentum)
            self.op_prediction = self.prediction(op_logits)

            # Initialize variables, i.e. weights and biases.
            self.op_init = tf.global_variables_initializer()
            
            # Summaries for TensorBoard and Save for model parameters.
            self.op_summary = tf.summary.merge_all()
            self.op_saver = tf.train.Saver(max_to_keep=5)
        
        self.graph.finalize()
    
    def inference(self, data, laplacians, dropout):
        """
        It builds the model, i.e. the computational graph, as far as
        is required for running the network forward to make predictions,
        i.e. return logits given raw data.

        data: size N x M
            N: number of signals (samples)
            M: number of vertices (features)
        training: we may want to discriminate the two, e.g. for dropout.
            True: the model is built for training.
            False: the model is built for evaluation.
        """
        # TODO: optimizations for sparse data
        logits = self._inference(data, laplacians, dropout)
        return logits
    
    def probabilities(self, logits):
        """Return the probability of a sample to belong to each class."""
        with tf.name_scope('probabilities'):
            probabilities = tf.nn.softmax(logits)
            return probabilities

    def prediction(self, logits):
        #"""Return the predicted classes."""
        """Return the predicted values."""
        with tf.name_scope('prediction'):
            prediction = tf.nn.softmax(logits)
            return prediction
            #return logits

    def loss(self, logits, labels, regularization):
        """Adds to the inference model the layers required to generate loss."""
        with tf.name_scope('loss'):
            with tf.name_scope('cross_entropy'):
                labels = tf.cast(labels, tf.float32)
                
                #softmax cross entropy for multilabel
                #for i in range(len(labels)):
                #    s = np.sum(labels[i])
                #    s = max(s,1)
                #    labels[i]=labels[i]/s
                #cross_entropy = tf.reduce_sum(tf.multiply(labels, -tf.log(tf.nn.softmax(logits))), axis=1)
                #cross_entropy = tf.reduce_mean(cross_entropy)
                
                #sigmoid cross entropy for multilabel
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
                cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1) )
            with tf.name_scope('regularization'):
                regularization *= tf.add_n(self.regularizers)
            loss = cross_entropy + regularization
                
            # Summaries for TensorBoard.
            tf.summary.scalar('loss/cross_entropy', cross_entropy)
            tf.summary.scalar('loss/regularization', regularization)
            tf.summary.scalar('loss/total', loss)
            with tf.name_scope('averages'):
                averages = tf.train.ExponentialMovingAverage(0.9)
                op_averages = averages.apply([cross_entropy, regularization, loss])
                tf.summary.scalar('loss/avg/cross_entropy', averages.average(cross_entropy))
                tf.summary.scalar('loss/avg/regularization', averages.average(regularization))
                tf.summary.scalar('loss/avg/total', averages.average(loss))
                with tf.control_dependencies([op_averages]):
                    loss_average = tf.identity(averages.average(loss), name='control')
            return loss, loss_average
    
    def training(self, loss, learning_rate, decay_steps, decay_rate=0.95, momentum=0.9):
        """Adds to the loss model the Ops required to generate and apply gradients."""
        with tf.name_scope('training'):
            # Learning rate.
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if decay_rate != 1:
                learning_rate = tf.train.exponential_decay(
                        learning_rate, global_step, decay_steps, decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)
            # Optimizer.
            if momentum == 0:
                #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            else:
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
            grads = optimizer.compute_gradients(loss)
            op_gradients = optimizer.apply_gradients(grads, global_step=global_step)
            # Histograms.
            for grad, var in grads:
                if grad is None:
                    print('warning: {} has no gradient'.format(var.op.name))
                else:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            # The op return the learning rate.
            with tf.control_dependencies([op_gradients]):
                op_train = tf.identity(learning_rate, name='control')
            return op_train

    # Helper methods.

    def _get_path(self, folder):
        path = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(path, '..', folder, self.dir_name)

    def _get_session(self, sess=None):
        """Restore parameters if no session given."""
        if sess is None:
            # tesnsorflow usage settings
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.400)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            sess = tf.Session(graph=self.graph)
            filename = tf.train.latest_checkpoint(self._get_path('checkpoints'))
            self.op_saver.restore(sess, filename)
        return sess

    def _weight_variable(self, shape, regularization=True):
        initial = tf.truncated_normal_initializer(0, 0.1)
        var = tf.get_variable('weights', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _bias_variable(self, shape, regularization=True):
        initial = tf.constant_initializer(0.1)
        var = tf.get_variable('bias', shape, tf.float32, initializer=initial)
        if regularization:
            self.regularizers.append(tf.nn.l2_loss(var))
        tf.summary.histogram(var.op.name, var)
        return var

    def _conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Fully connected


class fc1(base_model):
    def __init__(self):
        super().__init__()
    def _inference(self, x, dropout):
        W = self._weight_variable([NFEATURES, NCLASSES])
        b = self._bias_variable([NCLASSES])
        y = tf.matmul(x, W) + b
        return y

class fc2(base_model):
    def __init__(self, nhiddens):
        super().__init__()
        self.nhiddens = nhiddens
    def _inference(self, x, dropout):
        with tf.name_scope('fc1'):
            W = self._weight_variable([NFEATURES, self.nhiddens])
            b = self._bias_variable([self.nhiddens])
            y = tf.nn.relu(tf.matmul(x, W) + b)
        with tf.name_scope('fc2'):
            W = self._weight_variable([self.nhiddens, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.matmul(y, W) + b
        return y


# Convolutional


class cnn2(base_model):
    """Simple convolutional model."""
    def __init__(self, K, F):
        super().__init__()
        self.K = K  # Patch size
        self.F = F  # Number of features
    def _inference(self, x, dropout):
        with tf.name_scope('conv1'):
            W = self._weight_variable([self.K, self.K, 1, self.F])
            b = self._bias_variable([self.F])
#            b = self._bias_variable([1, 28, 28, self.F])
            x_2d = tf.reshape(x, [-1,28,28,1])
            y_2d = self._conv2d(x_2d, W) + b
            y_2d = tf.nn.relu(y_2d)
        with tf.name_scope('fc1'):
            y = tf.reshape(y_2d, [-1, NFEATURES*self.F])
            W = self._weight_variable([NFEATURES*self.F, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.matmul(y, W) + b
        return y

class fcnn2(base_model):
    """CNN using the FFT."""
    def __init__(self, F):
        super().__init__()
        self.F = F  # Number of features
    def _inference(self, x, dropout):
        with tf.name_scope('conv1'):
            # Transform to Fourier domain
            x_2d = tf.reshape(x, [-1, 28, 28])
            x_2d = tf.complex(x_2d, 0)
            xf_2d = tf.fft2d(x_2d)
            xf = tf.reshape(xf_2d, [-1, NFEATURES])
            xf = tf.expand_dims(xf, 1)  # NSAMPLES x 1 x NFEATURES
            xf = tf.transpose(xf)  # NFEATURES x 1 x NSAMPLES
            # Filter
            Wreal = self._weight_variable([int(NFEATURES/2), self.F, 1])
            Wimg = self._weight_variable([int(NFEATURES/2), self.F, 1])
            W = tf.complex(Wreal, Wimg)
            xf = xf[:int(NFEATURES/2), :, :]
            yf = tf.matmul(W, xf)  # for each feature
            yf = tf.concat([yf, tf.conj(yf)], axis=0)
            yf = tf.transpose(yf)  # NSAMPLES x NFILTERS x NFEATURES
            yf_2d = tf.reshape(yf, [-1, 28, 28])
            # Transform back to spatial domain
            y_2d = tf.ifft2d(yf_2d)
            y_2d = tf.real(y_2d)
            y = tf.reshape(y_2d, [-1, self.F, NFEATURES])
            # Bias and non-linearity
            b = self._bias_variable([1, self.F, 1])
#            b = self._bias_variable([1, self.F, NFEATURES])
            y += b  # NSAMPLES x NFILTERS x NFEATURES
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*NFEATURES, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*NFEATURES])
            y = tf.matmul(y, W) + b
        return y


# Graph convolutional


class fgcnn2(base_model):
    """Graph CNN with full weights, i.e. patch has the same size as input."""
    def __init__(self, L, F):
        super().__init__()
        #self.L = L  # Graph Laplacian, NFEATURES x NFEATURES
        self.F = F  # Number of filters
        _, self.U = graph.fourier(L)
    def _inference(self, x, dropout):
        # x: NSAMPLES x NFEATURES
        with tf.name_scope('gconv1'):
            # Transform to Fourier domain
            U = tf.constant(self.U, dtype=tf.float32)
            xf = tf.matmul(x, U)
            xf = tf.expand_dims(xf, 1)  # NSAMPLES x 1 x NFEATURES
            xf = tf.transpose(xf)  # NFEATURES x 1 x NSAMPLES
            # Filter
            W = self._weight_variable([NFEATURES, self.F, 1])
            yf = tf.matmul(W, xf)  # for each feature
            yf = tf.transpose(yf)  # NSAMPLES x NFILTERS x NFEATURES
            yf = tf.reshape(yf, [-1, NFEATURES])
            # Transform back to graph domain
            Ut = tf.transpose(U)
            y = tf.matmul(yf, Ut)
            y = tf.reshape(yf, [-1, self.F, NFEATURES])
            # Bias and non-linearity
            b = self._bias_variable([1, self.F, 1])
#            b = self._bias_variable([1, self.F, NFEATURES])
            y += b  # NSAMPLES x NFILTERS x NFEATURES
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*NFEATURES, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*NFEATURES])
            y = tf.matmul(y, W) + b
        return y


class lgcnn2_1(base_model):
    """Graph CNN which uses the Lanczos approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        self.L = L  # Graph Laplacian, M x M
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M, K = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Transform to Lanczos basis
            xl = tf.reshape(x, [-1, self.K])  # NM x K
            # Filter
            W = self._weight_variable([self.K, self.F])
            y = tf.matmul(xl, W)  # NM x F
            y = tf.reshape(y, [-1, M, self.F])  # N x M x F
            # Bias and non-linearity
            b = self._bias_variable([1, 1, self.F])
#            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y

class lgcnn2_2(base_model):
    """Graph CNN which uses the Lanczos approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        self.L = L  # Graph Laplacian, M x M
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Transform to Lanczos basis
            xl = tf.transpose(x)  # M x N
            def lanczos(x):
                return graph.lanczos(self.L, x, self.K)
            xl = tf.py_func(lanczos, [xl], [tf.float32])[0]
            xl = tf.transpose(xl)  # N x M x K
            xl = tf.reshape(xl, [-1, self.K])  # NM x K
            # Filter
            W = self._weight_variable([self.K, self.F])
            y = tf.matmul(xl, W)  # NM x F
            y = tf.reshape(y, [-1, M, self.F])  # N x M x F
            # Bias and non-linearity
#            b = self._bias_variable([1, 1, self.F])
            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y


class cgcnn2_2(base_model):
    """Graph CNN which uses the Chebyshev approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        self.L = graph.rescale_L(L, lmax=2)  # Graph Laplacian, M x M
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Transform to Chebyshev basis
            xc = tf.transpose(x)  # M x N
            def chebyshev(x):
                return graph.chebyshev(self.L, x, self.K)
            xc = tf.py_func(chebyshev, [xc], [tf.float32])[0]
            xc = tf.transpose(xc)  # N x M x K
            xc = tf.reshape(xc, [-1, self.K])  # NM x K
            # Filter
            W = self._weight_variable([self.K, self.F])
            y = tf.matmul(xc, W)  # NM x F
            y = tf.reshape(y, [-1, M, self.F])  # N x M x F
            # Bias and non-linearity
#            b = self._bias_variable([1, 1, self.F])
            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y


class cgcnn2_3(base_model):
    """Graph CNN which uses the Chebyshev approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        L = graph.rescale_L(L, lmax=2)  # Graph Laplacian, M x M
        self.L = L.toarray()
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Filter
            W = self._weight_variable([self.K, self.F])
            def filter(xt, k):
                xt = tf.reshape(xt, [-1, 1])  # NM x 1
                w = tf.slice(W, [k,0], [1,-1])  # 1 x F
                y = tf.matmul(xt, w)  # NM x F
                return tf.reshape(y, [-1, M, self.F])  # N x M x F
            xt0 = x
            y = filter(xt0, 0)
            if self.K > 1:
                xt1 = tf.matmul(x, self.L, b_is_sparse=True)  # N x M
                y += filter(xt1, 1)
            for k in range(2, self.K):
                xt2 = 2 * tf.matmul(xt1, self.L, b_is_sparse=True) - xt0  # N x M
                y += filter(xt2, k)
                xt0, xt1 = xt1, xt2
            # Bias and non-linearity
#            b = self._bias_variable([1, 1, self.F])
            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y


class cgcnn2_4(base_model):
    """Graph CNN which uses the Chebyshev approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        L = graph.rescale_L(L, lmax=2)  # Graph Laplacian, M x M
        L = L.tocoo()
        data = L.data
        indices = np.empty((L.nnz, 2))
        indices[:,0] = L.row
        indices[:,1] = L.col
        L = tf.SparseTensor(indices, data, L.shape)
        self.L = tf.sparse_reorder(L)
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Filter
            W = self._weight_variable([self.K, self.F])
            def filter(xt, k):
                xt = tf.transpose(xt)  # N x M
                xt = tf.reshape(xt, [-1, 1])  # NM x 1
                w = tf.slice(W, [k,0], [1,-1])  # 1 x F
                y = tf.matmul(xt, w)  # NM x F
                return tf.reshape(y, [-1, M, self.F])  # N x M x F
            xt0 = tf.transpose(x)  # M x N
            y = filter(xt0, 0)
            if self.K > 1:
                xt1 = tf.sparse_tensor_dense_matmul(self.L, xt0)
                y += filter(xt1, 1)
            for k in range(2, self.K):
                xt2 = 2 * tf.sparse_tensor_dense_matmul(self.L, xt1) - xt0  # M x N
                y += filter(xt2, k)
                xt0, xt1 = xt1, xt2
            # Bias and non-linearity
#            b = self._bias_variable([1, 1, self.F])
            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y


class cgcnn2_5(base_model):
    """Graph CNN which uses the Chebyshev approximation."""
    def __init__(self, L, F, K):
        super().__init__()
        L = graph.rescale_L(L, lmax=2)  # Graph Laplacian, M x M
        L = L.tocoo()
        data = L.data
        indices = np.empty((L.nnz, 2))
        indices[:,0] = L.row
        indices[:,1] = L.col
        L = tf.SparseTensor(indices, data, L.shape)
        self.L = tf.sparse_reorder(L)
        self.F = F  # Number of filters
        self.K = K  # Polynomial order, i.e. filter size (number of hopes)
    def _inference(self, x, dropout):
        with tf.name_scope('gconv1'):
            N, M = x.get_shape()  # N: number of samples, M: number of features
            M = int(M)
            # Transform to Chebyshev basis
            xt0 = tf.transpose(x)  # M x N
            xt = tf.expand_dims(xt0, 0)  # 1 x M x N
            def concat(xt, x):
                x = tf.expand_dims(x, 0)  # 1 x M x N
                return tf.concat([xt, x], axis=0)  # K x M x N
            if self.K > 1:
                xt1 = tf.sparse_tensor_dense_matmul(self.L, xt0)
                xt = concat(xt, xt1)
            for k in range(2, self.K):
                xt2 = 2 * tf.sparse_tensor_dense_matmul(self.L, xt1) - xt0  # M x N
                xt = concat(xt, xt2)
                xt0, xt1 = xt1, xt2
            xt = tf.transpose(xt)  # N x M x K
            xt = tf.reshape(xt, [-1,self.K])  # NM x K
            # Filter
            W = self._weight_variable([self.K, self.F])
            y = tf.matmul(xt, W)  # NM x F
            y = tf.reshape(y, [-1, M, self.F])  # N x M x F
            # Bias and non-linearity
#            b = self._bias_variable([1, 1, self.F])
            b = self._bias_variable([1, M, self.F])
            y += b  # N x M x F
            y = tf.nn.relu(y)
        with tf.name_scope('fc1'):
            W = self._weight_variable([self.F*M, NCLASSES])
            b = self._bias_variable([NCLASSES])
            y = tf.reshape(y, [-1, self.F*M])
            y = tf.matmul(y, W) + b
        return y


def bspline_basis(K, x, degree=3):
    """
    Return the B-spline basis.

    K: number of control points.
    x: evaluation points
       or number of evenly distributed evaluation points.
    degree: degree of the spline. Cubic spline by default.
    """
    if np.isscalar(x):
        x = np.linspace(0, 1, x)

    # Evenly distributed knot vectors.
    kv1 = x.min() * np.ones(degree)
    kv2 = np.linspace(x.min(), x.max(), K-degree+1)
    kv3 = x.max() * np.ones(degree)
    kv = np.concatenate((kv1, kv2, kv3))

    # Cox - DeBoor recursive function to compute one spline over x.
    def cox_deboor(k, d):
        # Test for end conditions, the rectangular degree zero spline.
        if (d == 0):
            return ((x - kv[k] >= 0) & (x - kv[k + 1] < 0)).astype(int)

        denom1 = kv[k + d] - kv[k]
        term1 = 0
        if denom1 > 0:
            term1 = ((x - kv[k]) / denom1) * cox_deboor(k, d - 1)

        denom2 = kv[k + d + 1] - kv[k + 1]
        term2 = 0
        if denom2 > 0:
            term2 = ((-(x - kv[k + d + 1]) / denom2) * cox_deboor(k + 1, d - 1))

        return term1 + term2

    # Compute basis for each point
    basis = np.column_stack([cox_deboor(k, degree) for k in range(K)])
    basis[-1,-1] = 1
    return basis


class cgcnn(base_model):
    """
    Graph CNN which uses the Chebyshev approximation.

    The following are hyper-parameters of graph convolutional layers.
    They are lists, which length is equal to the number of gconv layers.
        F: Number of features.
        K: List of polynomial orders, i.e. filter sizes or number of hopes.
        p: Pooling size.
           Should be 1 (no pooling) or a power of 2 (reduction by 2 at each coarser level).
           Beware to have coarsened enough.

    L: List of Graph Laplacians. Size M x M. One per coarsening level.

    The following are hyper-parameters of fully connected layers.
    They are lists, which length is equal to the number of fc layers.
        M: Number of features per sample, i.e. number of hidden neurons.
           The last layer is the softmax, i.e. M[-1] is the number of classes.
    
    The following are choices of implementation for various blocks.
        filter: filtering operation, e.g. chebyshev5, lanczos2 etc.
        brelu: bias and relu, e.g. b1relu or b2relu.
        pool: pooling, e.g. mpool1.
    
    Training parameters:
        num_epochs:    Number of training epochs.
        learning_rate: Initial learning rate.
        decay_rate:    Base of exponential decay. No decay with 1.
        decay_steps:   Number of steps after which the learning rate decays.
        momentum:      Momentum. 0 indicates no momentum.

    Regularization parameters:
        regularization: L2 regularizations of weights and biases.
        dropout:        Dropout (fc layers): probability to keep hidden neurons. No dropout with 1.
        batch_size:     Batch size. Must divide evenly into the dataset sizes.
        eval_frequency: Number of steps between evaluations.

    Directories:
        dir_name: Name for directories (summaries and model parameters).
    """
    def __init__(self, L, num_labels_per_image, F, F_0, K, p, M,train_laplacians, test_laplacians, val_laplacians,
                filter='chebyshev5', brelu='b1relu', pool='mpool1',
                num_epochs=20, learning_rate=0.1, decay_rate=0.95, decay_steps=None, momentum=0.9,
                regularization=0, dropout=0, batch_size=100, eval_frequency=200,
                dir_name=''):
        super().__init__(num_labels_per_image)
        # Verify the consistency w.r.t. the number of layers.
        assert len(L) >= len(F) == len(K) == len(p)
        assert np.all(np.array(p) >= 1)
        p_log2 = np.where(np.array(p) > 1, np.log2(p), 0)
        assert np.all(np.mod(p_log2, 1) == 0)  # Powers of 2.
        assert len(L) >= 1 + np.sum(p_log2)  # Enough coarsening levels for pool sizes.
        
        # Keep the useful Laplacians only. May be zero.
        M_0 = L[0].shape[0]
        j = 0
        self.L = []
        self.train_laplacians = []
        self.val_laplacians = []
        self.test_laplacians = []
        for pp in p:
            self.L.append(L[j])
            self.train_laplacians.append(train_laplacians[j])
            self.val_laplacians.append(val_laplacians[j])
            self.test_laplacians.append(test_laplacians[j])
            j += int(np.log2(pp)) if pp > 1 else 0
        L = self.L
        train_laplacians = self.train_laplacians
        val_laplacians = self.val_laplacians
        test_laplacians = self.test_laplacians

        # Print information about NN architecture.
        Ngconv = len(p)
        Nfc = len(M)
        print('NN architecture')
        print('  input: M_0 = {}'.format(M_0))
        for i in range(Ngconv):
            print('  layer {0}: cgconv{0}'.format(i+1))
            print('    representation: M_{0} * F_{1} / p_{1} = {2} * {3} / {4} = {5}'.format(
                    i, i+1, L[i].shape[0], F[i], p[i], L[i].shape[0]*F[i]//p[i]))
            F_last = F[i-1] if i > 0 else F_0
            print('    weights: F_{0} * F_{1} * K_{1} = {2} * {3} * {4} = {5}'.format(
                    i, i+1, F_last, F[i], K[i], F_last*F[i]*K[i]))
            if brelu == 'b1relu':
                print('    biases: F_{} = {}'.format(i+1, F[i]))
            elif brelu == 'b2relu':
                print('    biases: M_{0} * F_{0} = {1} * {2} = {3}'.format(
                        i+1, L[i].shape[0], F[i], L[i].shape[0]*F[i]))
        for i in range(Nfc):
            name = 'logits (softmax)' if i == Nfc-1 else 'fc{}'.format(i+1)
            print('  layer {}: {}'.format(Ngconv+i+1, name))
            print('    representation: M_{} = {}'.format(Ngconv+i+1, M[i]))
            M_last = M[i-1] if i > 0 else M_0 if Ngconv == 0 else L[-1].shape[0] * F[-1] // p[-1]
            print('    weights: M_{} * M_{} = {} * {} = {}'.format(
                    Ngconv+i, Ngconv+i+1, M_last, M[i], M_last*M[i]))
            print('    biases: M_{} = {}'.format(Ngconv+i+1, M[i]))
        
        # Store attributes and bind operations.
        self.L, self.F, self.K, self.p, self.M = L, F, K, p, M
        self.train_laplacians = train_laplacians
        self.val_laplacians = val_laplacians
        self.test_laplacians =  test_laplacians
        #if type(train_laplacians) is not np.ndarray:
        #    self.train_laplacians = np.asarray(self.train_laplacians)
        #if type(val_laplacians) is not np.ndarray:
        #    self.val_laplacians = np.asarray(self.val_laplacians)
        #if type(test_laplacians) is not np.ndarray:
        #    self.test_laplacians = np.asarray(self.test_laplacians)
        self.num_epochs, self.learning_rate = num_epochs, learning_rate
        self.decay_rate, self.decay_steps, self.momentum = decay_rate, decay_steps, momentum
        self.regularization, self.dropout = regularization, dropout
        self.batch_size, self.eval_frequency = batch_size, eval_frequency
        self.dir_name = dir_name
        self.filter = getattr(self, filter)
        self.brelu = getattr(self, brelu)
        self.pool = getattr(self, pool)
        
        # Build the computational graph.
        self.build_graph(M_0, F_0, M[-1], L)
        
    def filter_in_fourier(self, x, laplacians, Fout, K, U, W):
        # TODO: N x F x M would avoid the permutations
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        # Transform to Fourier domain
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        x = tf.matmul(U, x)  # M x Fin*N
        x = tf.reshape(x, [M, Fin, N])  # M x Fin x N
        # Filter
        x = tf.matmul(W, x)  # for each feature
        x = tf.transpose(x)  # N x Fout x M
        x = tf.reshape(x, [N*Fout, M])  # N*Fout x M
        # Transform back to graph domain
        x = tf.matmul(x, U)  # N*Fout x M
        x = tf.reshape(x, [N, Fout, M])  # N x Fout x M
        return tf.transpose(x, perm=[0, 2, 1])  # N x M x Fout

    def fourier(self, x, laplacians, Fout, K):
        laplacians = laplacians[0]
        #assert K == L.shape[0]  # artificial but useful to compute number of parameters
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        _, U = graph.fourier(laplacians)
        U = tf.constant(U.T, dtype=tf.float32)
        # Weights
        W = self._weight_variable([M, Fout, Fin], regularization=False)
        return self.filter_in_fourier(x, laplacians, Fout, K, U, W)

    def spline(self, x, L, Fout, K):
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Fourier basis
        lamb, U = graph.fourier(L)
        U = tf.constant(U.T, dtype=tf.float32)  # M x M
        # Spline basis
        B = bspline_basis(K, lamb, degree=3)  # M x K
        #B = bspline_basis(K, len(lamb), degree=3)  # M x K
        B = tf.constant(B, dtype=tf.float32)
        # Weights
        W = self._weight_variable([K, Fout*Fin], regularization=False)
        W = tf.matmul(B, W)  # M x Fout*Fin
        W = tf.reshape(W, [M, Fout, Fin])
        return self.filter_in_fourier(x, L, Fout, K, U, W)

    #def chebyshev2(self, x, L, Fout, K):
    def chebyshev2(self, x, laplacians, Fout, K):
        """
        Filtering with Chebyshev interpolation
        Implementation: numpy.
        
        Data: x of size N x M x F
            N: number of signals
            M: number of vertices
            F: number of features per signal per vertex
        """
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian. Copy to not modify the shared L.
        laplacians = scipy.sparse.csr_matrix(laplacians)
        laplacians = graph.rescale_L(laplacians, lmax=2)
        # Transform to Chebyshev basis
        x = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        x = tf.reshape(x, [M, Fin*N])  # M x Fin*N
        def chebyshev(x):
            return graph.chebyshev(laplacians, x, K)
        x = tf.py_func(chebyshev, [x], [tf.float32])[0]  # K x M x Fin*N
        x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def chebyshev5(self, x, laplacians, Fout, K):
        #laplacians = laplacians[0]
        N, M, Fin = x.get_shape()
        N, M, Fin = int(N), int(M), int(Fin)
        # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.
        #laplacians = scipy.sparse.csr_matrix(laplacians)
        #laplacians = graph.rescale_L(laplacians, lmax=2)
        # todo  rescale lapalcians

        #laplacians = laplacians.tocoo()
        #indices = np.column_stack((laplacians.row, laplacians.col))
        #laplacians = tf.SparseTensor(indices, laplacians.data, laplacians.shape)
        #laplacians = tf.sparse_reorder(laplacians)
        # Transform to Chebyshev basis
        x0 = x  # N * M * Fin
        #x0 = tf.transpose(x, perm=[1, 2, 0])  # M x Fin x N
        #x0 = tf.reshape(x0, [M, Fin*N])  # M x Fin*N
        x = tf.expand_dims(x0, 0)  # 1 x N x M x FinN
        def concat(x, x_):
            #x_ = tf.expand_dims(x_, 0)  # 1 x M x Fin*N
            #return tf.concat([x, x_], axis=0)  # K x M x Fin*N
            x_ = tf.expand_dims(x_, 0)  # 1 x N x M x Fin
            return tf.concat([x, x_], axis=0)  # K x N x M x Fin
        if K > 1:
            #x1 = tf.sparse_tensor_dense_matmul(laplacians, x0)
            x1 = tf.matmul(laplacians, x0)       # N x M x Fin
            x = concat(x, x1)
        for k in range(2, K):
            #x2 = 2 * tf.sparse_tensor_dense_matmul(laplacians, x1) - x0  # M x Fin*N
            x2 = 2 * tf.matmul(laplacians, x1) - x0   # N x M x Fin
            x = concat(x, x2)
            x0, x1 = x1, x2
        #x = tf.reshape(x, [K, M, Fin, N])  # K x M x Fin x N
        #x = tf.transpose(x, perm=[3,1,2,0])  # N x M x Fin x K
        x = tf.transpose(x, perm=[1,2,3,0])  # N x M x Fin x K
        x = tf.reshape(x, [N*M, Fin*K])  # N*M x Fin*K
        # Filter: Fin*Fout filters of order K, i.e. one filterbank per feature pair.
        W = self._weight_variable([Fin*K, Fout], regularization=False)
        x = tf.matmul(x, W)  # N*M x Fout
        return tf.reshape(x, [N, M, Fout])  # N x M x Fout

    def b1relu(self, x):
        """Bias and ReLU. One bias per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, 1, int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def b2relu(self, x):
        """Bias and ReLU. One bias per vertex per filter."""
        N, M, F = x.get_shape()
        b = self._bias_variable([1, int(M), int(F)], regularization=False)
        return tf.nn.relu(x + b)

    def mpool1(self, x, p):
        """Max pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.max_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            #tf.maximum
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def apool1(self, x, p):
        """Average pooling of size p. Should be a power of 2."""
        if p > 1:
            x = tf.expand_dims(x, 3)  # N x M x F x 1
            x = tf.nn.avg_pool(x, ksize=[1,p,1,1], strides=[1,p,1,1], padding='SAME')
            return tf.squeeze(x, [3])  # N x M/p x F
        else:
            return x

    def fc(self, x, Mout, relu=True):
        """Fully connected layer with Mout features."""
        N, Min = x.get_shape()
        W = self._weight_variable([int(Min), Mout], regularization=True)
        b = self._bias_variable([Mout], regularization=True)
        x = tf.matmul(x, W) + b
        return tf.nn.relu(x) if relu else x

    def _inference(self, x, laplacians, dropout):
        # Graph convolutional layers.
        if len(x.shape) is 2:
            x = tf.expand_dims(x, 2)  # N x M x F(=1)
            
        for i in range(len(self.p)):
            with tf.variable_scope('conv{}'.format(i+1)):
                with tf.name_scope('filter'):
                    #x = self.filter(x, self.L[i], self.F[i], self.K[i])
                    x = self.filter(x, laplacians[i], self.F[i], self.K[i])
                with tf.name_scope('bias_relu'):
                    x = self.brelu(x)
                with tf.name_scope('pooling'):
                    x = self.pool(x, self.p[i])
        
        # Fully connected hidden layers.
        N, M, F = x.get_shape()
        x = tf.reshape(x, [int(N), int(M*F)])  # N x M
        for i,M in enumerate(self.M[:-1]):
            with tf.variable_scope('fc{}'.format(i+1)):
                x = self.fc(x, M)
                x = tf.nn.dropout(x, dropout)
        
        # Logits linear layer, i.e. softmax without normalization.
        with tf.variable_scope('logits'):
            x = self.fc(x, self.M[-1], relu=False)
        return x
