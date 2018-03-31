import tensorflow as tf
import numpy as np
import os
import shutil
from copy import copy, deepcopy

class GraphInfo():
    def __init__(self,id,layers,ls_units):
        """
        every instance of Graphinfo will create a new model directory
        :param id: id of the model
        :param layers: no. of layers including input and output int8
        :param ls_units: a list of no. of units in each layer
        """
        try:
            assert layers == len(ls_units)
        except AssertionError as e:
            e.args = e.args + ("no. of layer inconsistent, check units list. Be sure to include"
                            " input and output layers",)
            raise
        else:
            # no. of hidden layers in the model
            self.layers = layers

            # no. of units in each layer
            self.units = ls_units

            # id of the model
            self.id = id

    def insertLayer(self,after_layer,inplace = False):

        """
        Inserts a layer after the location given by after_layer
        :param after_layer: index of layer after which the new layer is to be inserted
        :param inplace: if true, it will modify the object inplace, if false it will create a new object and modify
        :return: None if inplace is True, modified temp object if inplace is False
        """
        try:
            assert (after_layer<self.layers - 1) and (after_layer > 0)
        except AssertionError as err:
            err.args = err.args + (" after_layer index out of bounds",)
            raise
        else:
            if inplace:
                self.layers += 1
                prev_layer_units = self.units[after_layer]
                self.units.insert(after_layer,prev_layer_units)
                return None

            else:
                temp_ginfo = deepcopy(self)
                temp_ginfo.id += 1
                temp_ginfo.layers += 1
                prev_layer_units = temp_ginfo.units[after_layer]
                temp_ginfo.units.insert(after_layer, prev_layer_units)
                return temp_ginfo

    def addUnits(self,layer_index,units, inplace = False):
        """
        :param layer_index: index of layer where to add units
        :param units: no. of units to add in this layer
        :return: a GraphInfo object, the existing object is not modified
        """
        temp_ginfo = deepcopy(self)
        try:
            assert units > 0
        except AssertionError as err:
            err.args = err.args + ("Cannot shrink the network, enter a positive number.",)
            raise
        else:
            temp_ginfo.units[layer_index] += units
            return temp_ginfo

    def printInfo(self):
        print("ID: ",self.id,"\tunits: ",self.units,"\tlayers: ",self.layers)


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
            w0 = np.random.normal(loc = 0.0, scale = 0.1, size=[784,self.ls_units[1]])
            w1 = np.random.normal(loc = 0.0, scale = 0.1, size = [self.ls_units[1],10])
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

            self.new_kernel = sess.run(self.kernel)
            for arr in self.new_kernel:
                print(arr.shape)


    def write_weights(self):
        temp_count = 0
        for arr in self.new_kernel:
            temp_fname = temp_fname = self.dir +"/w"+str(temp_count)+".csv"
            os.remove(temp_fname)
            np.savetxt(temp_fname,arr,delimiter=",")
            temp_count += 1


def net2Wider(prev_graphinfo,next_graphinfo):
    """
    Will write modified weights to the sub-directory of new graph info
    """
    next_layers = next_graphinfo.layers
    next_units = next_graphinfo.units

    prev_layers = prev_graphinfo.layers
    prev_units = prev_graphinfo.units

    prev_dir = prev_graphinfo.dest_dir
    target_dir = next_graphinfo.dest_dir

    try:
        assert prev_layers == next_layers
    except AssertionError:
        print("Layers should remain same in net2wider")
        return None
    else:
        diff = [b - a for a,b in zip(prev_units,next_units)]

        # fetch the weight matrices in all layers
        weights = [0]*(len(diff)-1)
        for layer,_ in enumerate(weights):
            temp_fname = prev_dir+"/w"+str(layer)+".csv"
            weights[layer] = np.genfromtxt(temp_fname,delimiter=",")

        # when new units are added in layer n, weight matrices of layer(n-1) and layer(n) will be updated.
        for layer,i in enumerate(diff):
            if i > 0:
                # updating the weights in layer(n-1)
                # new weight matrix will have new columns and existing columns are left undisturbed.
                total_cols = weights[layer-1].shape[1]
                print(total_cols)
                rand_col_idx = np.random.choice(total_cols,size = i, replace = True)
                rand_cols = weights[layer-1][:,rand_col_idx]
                weights[layer-1] = np.append(weights[layer-1],rand_cols,axis = 1)

                # updating the weights in layer(n)
                # this new weight matrix will have new row and existing rows will be updated
                count_dict = dict()
                for idx in rand_col_idx:
                    if idx in count_dict:
                        count_dict[idx] += 1
                    else:
                        count_dict[idx] = 2
                for idx, count in count_dict.items():
                    weights[layer][idx,:] = weights[layer][idx,:]/float(count)

                # build new weight matrix
                for idx in rand_col_idx:
                    new_row = weights[layer][idx,:]
                    weights[layer] = np.vstack((weights[layer],new_row))

        # write the updated weight matrices to the target directory
        for layer,_ in enumerate(weights):
            temp_fname = target_dir+"/w"+str(layer)+".csv"
            np.savetxt(temp_fname,weights[layer],delimiter=",")


def net2Deeper(prev_graphinfo, next_graphinfo):
    """
    :param prev_graphinfo: as the name implies
    :param next_graphinfo: as the name implies
    """
    next_layers = next_graphinfo.layers
    next_units = next_graphinfo.units

    prev_layers = prev_graphinfo.layers
    prev_units = prev_graphinfo.units

    prev_dir = prev_graphinfo.dest_dir
    target_dir = next_graphinfo.dest_dir

    try:
        assert prev_layers != next_layers
    except AssertionError:
        print("Wrong method called...no.of layers must be different")
        return None
    else:
        diff = next_layers - prev_layers
        # now figure where to insert this hidden layer.


def main():
    history = History()
    graphinfo = history.get_graphinfo(model_id = 1)
    model1 = BuildModel(graphinfo)
    model1.train(5)
    model1.write_weights()
    #next_graphinfo = GraphInfo(id = 2,layers = 3, ls_units = [784,6,10])
    #net2Wider(graphinfo,next_graphinfo)


main()




