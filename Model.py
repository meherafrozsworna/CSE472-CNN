import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support as score

print("=====================================================================")
print("==========================  mnist dataset ===========================================")
print("=====================================================================")

def loadMNIST( data_file , label_file ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize
    data = np.fromfile(data_file , dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )
    labels = np.fromfile(label_file, dtype='ubyte')[2 * intType.itemsize:]

    return data, labels

x_train, y_train = loadMNIST( './MNIST/train-images.idx3-ubyte' , './MNIST/train-labels.idx1-ubyte' )  # ./Toy Dataset/trainNN.txt
x_test, y_test = loadMNIST( './MNIST/t10k-images.idx3-ubyte' , './MNIST/t10k-labels.idx1-ubyte' )

x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

x_train = x_train[0:1000]
x_test = x_test[0:1000]
y_train = y_train[0:1000]
y_test = y_test[0:1000]

x_test_full = x_test
y_test_full = y_test

indices = np.random.permutation(x_test_full.shape[0])              # permute and split training data in
count = x_test.shape[0]//2
test_idx, validation_idx = indices[:count], indices[count:]     # training and validation sets
x_test, validation_set = x_test_full[test_idx, :], x_test_full[validation_idx, :]
y_test, validation_labels = y_test_full[test_idx], y_test_full[validation_idx]

label_train = y_train
label_test = y_test
label_valid = validation_labels

print("Encoding ")
y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)
validation_labels = pd.get_dummies(validation_labels)


class Convolution_layer():
    def __init__(self, number_of_output_channels, filter_dimension, stride, padding):
        self.number_of_output_channels = number_of_output_channels
        self.filter_dimension = filter_dimension
        self.weight_scale = 0.01
        self.weights = None  #
        self.flag = 0
        self.stride = stride
        self.padding = padding
        self.input = None

    def paddingOperation(self,input,padding):
        out = None
        if padding > 0:
            out =  np.pad(input,[(0,0),(0,0),(padding,padding),(padding,padding)])
        elif padding == 0:
            out = input
        else :
            out = input[:, :, padding:-padding, padding:-padding]
        return out

    def forward(self,input):
        if self.flag == 0 :
            self.flag = 1
            c = input.shape[1]
            self.bias = np.random.randn(self.number_of_output_channels, 1) * np.sqrt(
                2.0 / (filter_dimension * filter_dimension * c))
            self.weights = np.random.randn(self.number_of_output_channels,c,filter_dimension,filter_dimension) *np.sqrt(2.0/(filter_dimension*filter_dimension*c))

        self.input = input
        mat_padded = self.paddingOperation(input,self.padding)
        out_dim = (mat_padded.shape[-1] - self.filter_dimension) // self.stride +1
        out = np.zeros((mat_padded.shape[0], self.number_of_output_channels,out_dim,out_dim))
        for i in range(out_dim):
            for j in range(out_dim):
                for k in range(self.number_of_output_channels):
                    out[:,k,i,j] = np.sum(
                        mat_padded[:,:,i*self.stride:i*self.stride + self.filter_dimension,
                        j * self.stride: j * self.stride + self.filter_dimension]
                        * self.weights[k],axis = (1,2,3)) + self.bias[k]

        return out

    def backward(self,dout, learning_rate):
        dim = (dout.shape[-1] -1 )*self.stride + 1
        mat_padded = self.paddingOperation(self.input, self.padding)
        out_n = np.zeros((dout.shape[0], dout.shape[1], dim , dim))
        out_n[:,:,0::self.stride, 0::self.stride] = dout
        df = np.zeros(self.weights.shape)

        for i in range(self.filter_dimension):
            for j in range(self.filter_dimension):
                for k in range(self.weights.shape[1]):
                    df[:,k,i,j] = np.average(np.sum(mat_padded[:,[k],i:i+dim,j:j + dim] *out_n , axis = (2,3)),axis = 0)

        db = np.average(np.sum(dout,axis = (2,3)),axis = 0).reshape(-1,1)
        padding_out = self.paddingOperation(out_n, self.filter_dimension - 1 -self.padding)
        new_filter = np.flip(self.weights, axis=(2,3))
        dx = np.zeros(self.input.shape)

        for i in range(self.input.shape[2]):
            for j in range(self.input.shape[3]):
                for k in range(self.input.shape[1]):
                    dx[:,k,i,j] = np.sum(padding_out[:,:,i:i+self.filter_dimension, j:j+self.filter_dimension] *
                                         new_filter[:,k],axis = (1,2,3))

        self.weights -= learning_rate * df *0.0001
        self.bias -= learning_rate*db * 0.0001

        return  dx


class Activation:
    def __init__(self):
        self.input = None

    def forward(self, input):
        self.input = input
        out = np.maximum(input, 0)
        return out

    def backward(self, dout, learning_rate):
        dx = dout * (self.input > 0)
        return dx


class Max_Pooling_layer:
    def __init__(self, filter_dimension, stride):
        self.filter_dimension = filter_dimension
        self.stride = stride
        self.input = None

    def forward(self, input):
        self.input = input

        N, C, H, W = input.shape
        out_H = 1 + (H - self.filter_dimension) // self.stride
        out_W = 1 + (W - self.filter_dimension) // self.stride

        out = np.zeros((N, C, out_H, out_W))

        for i in range(N):
            curr_out = np.zeros((C, out_H * out_W))
            c = 0
            for j in range(0, H - self.filter_dimension + 1, self.stride):
                for k in range(0, W - self.filter_dimension + 1, self.stride):
                    curr_region = input[i, :, j:j + self.filter_dimension, k:k + self.filter_dimension].reshape(C, self.filter_dimension * self.filter_dimension)
                    curr_max_pool = np.max(curr_region, axis=1)
                    curr_out[:, c] = curr_max_pool
                    c += 1
            out[i, :, :, :] = curr_out.reshape(C, out_H, out_W)

        return out

    def backward(self, dout, learning_rate):
        x = self.input
        stride = self.stride

        N, C, H, W = self.input.shape
        _, _, out_H, out_W = dout.shape

        dx = np.zeros_like(x)

        for i in range(N):
            curr_dout = dout[i, :].reshape(C, out_H * out_W)
            c = 0
            for j in range(0, H - self.filter_dimension + 1, stride):
                for k in range(0, W - self.filter_dimension + 1, stride):
                    curr_region = x[i, :, j:j + self.filter_dimension, k:k + self.filter_dimension].reshape(C, self.filter_dimension * self.filter_dimension)
                    curr_max_idx = np.argmax(curr_region, axis=1)
                    curr_dout_region = curr_dout[:, c]
                    curr_dpooling = np.zeros_like(curr_region)
                    curr_dpooling[np.arange(C), curr_max_idx] = curr_dout_region
                    dx[i, :, j:j + self.filter_dimension, k:k + self.filter_dimension] = curr_dpooling.reshape(C, self.filter_dimension, self.filter_dimension)
                    c += 1

        return dx


class Fully_connected_layer:
    def __init__(self, output_size):
        self.weights = None
        self.flag = 0
        self.bias = np.random.randn(output_size, 1)*0.001
        self.output_size = output_size
        # self.input_size = input_size
        self.input = None

    def forward(self, input):
        if self.flag == 0:
            input_size = input.shape[0]
            self.weights = np.random.randn( self.output_size, input_size)*0.001
            self.flag = 1
        self.input = input
        out = np.dot(self.weights, input) + self.bias #matmul (input*weigt)
        self.out = out
        return out

    def backward(self, dout, learning_rate):
        weights_gradient = np.dot(self.input, dout)
        input_gradient = np.dot(dout , self.weights)
        self.weights -= 0.0001*learning_rate * weights_gradient.T
        self.bias -= 0.0001 *learning_rate * np.array(np.mean(dout.T, axis=1)).reshape(self.output_size,1) #sum to mean
        return input_gradient



class Flattening_layer:
    def __init__(self):
        self.input = None
        self.shape = None

    def forward(self, input):
        self.input = input
        self.shape = input.shape
        c,n,h,w = input.shape
        x = input.reshape(c,n*h*w)
        out = x.T
        return out

    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.shape)


class Softmax:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        self.input = input
        input -= np.max(input,axis = -1 , keepdims = True)
        temp = np.exp(input)
        s = np.sum(temp, axis = 1 , keepdims = True)
        s[s == 0.0] = 1

        self.output = temp/s
        self.output = self.output.T
        return self.output

    def backward(self, output_gradient, learning_rate):
        out = self.output - output_gradient
        return out

    def loss(self, input_data , y):
        x = np.multiply(np.log(input_data), y)
        s = -np.sum(np.array(x))/input_data.shape[0]
        return s



class Convolutional_Model(object):
    def __init__(self):
        self.layers = []
    def add_layer(self,layer):
        self.layers.append(layer)

    def forward(self, input):
        for i in range(len(self.layers)):
            input = self.layers[i].forward(input)
        return input
    def backward(self, dout, learning_rate):
        for i in range(len(self.layers)):
            dout = self.layers[len(self.layers) - i-1].backward(dout,learning_rate)
        return dout

    def create_minibatch(self, x, y, batch_size=32):
        mini_batches = []
        num_examples = x.shape[0]
        num_batches = 2
        i = 0
        for i in range(num_batches):
            # random_indices = np.random.choice(num_examples, size=batch_size, replace=False)
            # x_mini = x[random_indices, :]
            # y_mini = y.iloc[random_indices , :]
            i  = random.randint(0, x.shape[0]- batch_size)
            x_mini = x[i :(i + batch_size), :]
            y_mini = y[i :(i + batch_size)  ]
            mini_batches.append((x_mini, y_mini))
        return mini_batches

    def loss(self, input_data , y ):
        x = np.multiply(np.log(input_data), y)
        # print(x.shape)
        s = -np.sum(np.array(x))/ input_data.shape[0]
        # print(s)
        return s

    def train(self, x, y, lr=0.001, batch_size=32, epochs=5):
        mini_batches = self.create_minibatch(x, y, batch_size)
        print('Splitted the training set into {} mini batches. \n'.format(len(mini_batches)))

        loss_history = []
        epochs = 5
        for epoch in range(epochs):
            input = x
            print('Epoch {}/{}: \n'.format(epoch + 1, epochs))
            for mini_batch in mini_batches:
                #print("?????????????????????????????????????????????????????????????????????????????????????????")
                x_mini, y_mini = mini_batch
                input = self.forward(x_mini)  #input will use for softmax loss
                dout = y_mini
                dout = self.backward(dout , lr)

            """ loss using validation set """
            # validation_set, validation_labels
            #
            print("Validation : ")
            validation_prediction = self.forward(validation_set)
            loss = self.loss(validation_prediction, validation_labels)
            print("Loss : ", loss)
            loss_history.append(loss)
            acc = self.accuracy(label_valid, validation_prediction)
            print("accuracy : ",acc*100)
            # precision, recall, fscore, support = score(label_valid, validation_prediction)
            # print("f1 score : ", fscore)

        return loss_history

    # def accuracy(self, actual, pred):
    #     accuracy = accuracy_score(y_true=actual, y_pred=pred)
    #     return accuracy
    def accuracy(self,y_actual , y_pred):
        predictions = np.argmax(y_pred, axis=1)
        accuracy = np.mean(predictions == y_actual)

        return accuracy

    def testing(self,x,y):
        test_pred = self.forward(validation_set)
        loss = self.loss(test_pred, y)
        print("Loss : ", loss)
        acc = self.accuracy(label_test, test_pred)
        print("accuracy : ", acc * 100)
        # precision, recall, fscore, support = score(label_test, test_pred)
        # print("f1 score : ",fscore)


steps_file = open('./input.txt')
Lines = steps_file.readlines()
model = Convolutional_Model()

for line in Lines:
    command = line.split()
    # print(command[0])
    if command[0] == "Conv":
        output_channels_number = int(command[1])
        filter_dimension = int(command[2])
        stride = int(command[3])
        padding = int(command[4])
        # input_channels = x_train.shape[1]   #input.shape[1]
        c = Convolution_layer(output_channels_number,filter_dimension,stride,padding)
        model.add_layer(c)
    elif command[0] =="ReLU":
        r = Activation()
        model.add_layer(r)
    elif command[0] == "Pool":
        filter_dimension = int(command[1])
        stride = int(command[2])
        p = Max_Pooling_layer(filter_dimension,stride)
        model.add_layer(p)
    elif command[0] == "FC":
        f = Flattening_layer()

        output_size = int(command[1])
        # input_size = input.shape[0]
        fc = Fully_connected_layer(output_size)
        model.add_layer(f)
        model.add_layer(fc)
    elif command[0] == "Flat":
        f = Flattening_layer()
        model.add_layer(f)
    elif command[0] == "Softmax":
        sf = Softmax()
        model.add_layer(sf)

losses = model.train(x_train, y_train, epochs=5)

print("TEST : ")
model.testing(x_test,y_test)



#
# print("=====================================================================")
# print("==========================  cifr10 dataset ===========================================")
# print("=====================================================================")
# #cifr10 dataset
# from tensorflow.keras.datasets import cifar10
#
# file = "Conv 32 3 1 0,ReLU,Conv 32 3 1 0,ReLU,Pool 2 2,Conv 64 3 1 0,Conv 64 3 1 0,Pool 2 2,FC 10,Softmax"
# Lines = file.split(',')
# model = ConvNet()
#
# for line in Lines:
#     command = line.split()
#     # print(command[0])
#     if command[0] == "Conv":
#         output_channels_number = int(command[1])
#         filter_dimension = int(command[2])
#         stride = int(command[3])
#         padding = int(command[4])
#         # input_channels = x_train.shape[1]   #input.shape[1]
#         c = Convolution_layer(output_channels_number,filter_dimension,stride,padding)
#         model.add_layer(c)
#     elif command[0] =="ReLU":
#         r = Activation()
#         model.add_layer(r)
#     elif command[0] == "Pool":
#         filter_dimension = int(command[1])
#         stride = int(command[2])
#         p = Max_Pooling_layer(filter_dimension,stride)
#         model.add_layer(p)
#     elif command[0] == "FC":
#         f = Flattening_layer()
#
#         output_size = int(command[1])
#         # input_size = input.shape[0]
#         fc = Fully_connected_layer(output_size)
#         model.add_layer(f)
#         model.add_layer(fc)
#     elif command[0] == "Flat":
#         f = Flattening_layer()
#         model.add_layer(f)
#     elif command[0] == "Softmax":
#         sf = Softmax()
#         model.add_layer(sf)
#
#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# # x_train = np.expand_dims(x_train, axis=1)
# # x_test = np.expand_dims(x_test, axis=1)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[3],x_train.shape[1],x_train.shape[2])
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[3],x_test.shape[1],x_test.shape[2])
# print(x_train.shape)
#
# y_train = y_train.reshape(y_train.shape[0])
# y_test = y_test.reshape(y_test.shape[0])
#
# x_train = x_train[0:1000]
# x_test = x_test[0:1000]
# y_train = y_train[0:1000]
# y_test = y_test[0:1000]
#
# x_test_full = x_test
# y_test_full = y_test
#
# indices = np.random.permutation(x_test_full.shape[0])              # permute and split training data in
# count = x_test.shape[0]//2
# test_idx, validation_idx = indices[:count], indices[count:]     # training and validation sets
# x_test, validation_set = x_test_full[test_idx, :], x_test_full[validation_idx, :]
# y_test, validation_labels = y_test_full[test_idx], y_test_full[validation_idx]
#
# label_train = y_train
# label_test = y_test
# label_valid = validation_labels
#
# print("Encoding ")
# print(y_train.reshape(y_train.shape[0]).shape)
# y_train = pd.get_dummies(y_train)
# print(y_train.shape)
# y_test = pd.get_dummies(y_test)
# validation_labels = pd.get_dummies(validation_labels)
# print(validation_labels.shape)
#
# model2 = Convolutional_Model()
# model2.train(x_train, y_train, label_valid ,validation_labels,epochs=5)
#
# print("TEST : ")
# model2.testing(x_test,y_test, label_test)
