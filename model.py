import tensorflow as tf
import numpy as np
import itertools
import time
import os
import shutil

class GraphInfo():
    def __init__(self,id,layers,ls_units):
        """
        every instance of Graphinfo will create a new model directory
        :param id: id of the model
        :param layers: no. of layers including input and output int8
        :param ls_units: a list of no. of units in each layer
        """
        # no. of hidden layers in the model
        self.layers = layers

        # no. of units in each layer
        self.units = ls_units

        # id of the model
        self.id = id

        self.dest_dir = "history/model"+str(self.id)
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        else:
            shutil.rmtree(self.dest_dir, ignore_errors=True)
            os.makedirs(self.dest_dir)


class History():
    def __init__(self):
        """
        first instance will have a default config of
            1 layer and 4 hidden units in that layer
        """
        self.history = dict()
        graph1 = GraphInfo(id = 1, layers = 3, ls_units = [784,4,10])
        self.add_graphinfo(graph1)

    def add_graphinfo(self,graph_info):
        """
        adds model_info to history
        """
        id = graph_info.id
        if id in self.history:
            print("model already exists in history")
        else:
            self.history[id] = graph_info

    def get_graphinfo(self,model_id):
        """
        return None if no GraphInfo object exists in history
        :param model_id: id of model to search for in history
        :return: a GraphInfo object who's model_id = model_id
        """
        return self.history[model_id]


class Data():
    def __init__(self,x_fname,y_fname):
        self.X = np.genfromtxt(x_fname,delimiter=",").astype(np.float32).reshape(-1,784)
        self.Y = np.genfromtxt(y_fname,delimiter=",").astype(np.float32).reshape(-1,10)

class BuildModel():
    def __init__(self,graph_info):
        """
        reads the graph structure form graph_info object,
        retrieves weights from sub-directory
        if model_id = 1, initialize weights inplace, or else extract from csv files
        """
        self.id = graph_info.id
        self.layers = graph_info.layers
        self.ls_units = graph_info.units
        self.dir = graph_info.dest_dir

        # if id = 1 write random weights to directory
        if self.id == 1:
            w0 = np.random.normal(loc = 0.0, scale = 0.1, size=[784,self.ls_units[0]])
            w1 = np.random.normal(loc = 0.0, scale = 0.1, size = [self.ls_units[0],10])
            np.savetxt(self.dir+"/w0.csv",w0,delimiter=",")
            np.savetxt(self.dir+"/w1.csv",w1,delimiter= ",")

        # retrieve weights from sub-directory
        self.weights = [0]*(self.layers-1)
        for i in range(self.layers-1):
            temp_fname = self.dir +"/w"+str(i)+".csv"
            self.weights[i] = tf.constant(np.genfromtxt(temp_fname,delimiter=",").astype(np.float32))

    def forward(self,x):
        self.kernel = [0]*(self.layers-1)
        l = [0]*(self.layers-1)
        l_relu = [0]*(self.layers)
        l_relu[0] = x
        for i in range(self.layers-1):
            with tf.variable_scope("layer"+str(i)):
                self.kernel[i] = tf.get_variable(name = "weight"+str(i),
                                                 initializer=self.weights[i],trainable=True)
                l[i] = tf.matmul(l_relu[i],self.kernel[i],name = "matmul")
                l_relu[i+1] = tf.nn.relu(l[i])
        prediction = tf.nn.softmax(l[self.layers-2])

        return prediction,l[self.layers-2]

    def train(self,epochs):
        x = tf.placeholder(tf.float32,shape = [None,784],name = "input_img")
        y_true = tf.placeholder(tf.float32,shape = [None,10], name = "true_cls")

        y_true_cls = tf.argmax(y_true)

        pred,logits = self.forward(x)

        pred_cls = tf.argmax(pred)
        acc = tf.reduce_sum(tf.cast(tf.equal(y_true_cls,pred_cls),tf.int8),name = "accuracy")

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels =y_true, logits =logits,name = "cost"))
        opt = tf.train.AdamOptimizer(1e-2).minimize(loss = cost,name = "Optimizer")

        # train the network, for every epoch display the accuracy on test set.
        with tf.Session() as sess:
            train_data = Data(x_fname=r"D:\data\mnist\x_train.csv", y_fname=r"D:\data\mnist\y_train.csv")
            test_data = Data(x_fname=r"D:\data\mnist\x_test.csv", y_fname=r"D:\data\mnist\y_test.csv")

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(epochs):

                train_dict = {x:train_data.X, y_true:train_data.Y}
                train_acc,loss,_ = sess.run([acc,cost,opt],feed_dict=train_dict)

                test_dict = {x: test_data.X, y_true: test_data.Y}
                test_acc = sess.run(acc,feed_dict=test_dict)

                print("Epoch: ",i,"\tcost: ",loss,"\ttrain_acc: ",train_acc,"\ttest_acc: ",test_acc)



def main():
    history = History()
    graphinfo = history.get_graphinfo(model_id = 1)
    model1 = BuildModel(graphinfo)
    model1.train(5)

main()




