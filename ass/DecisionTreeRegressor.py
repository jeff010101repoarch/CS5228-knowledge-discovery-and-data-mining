import numpy as np
import os
import json
import operator

class MyDecisionTreeRegressor():
    def __init__(self, max_depth=5, min_samples_split=2):
        '''
        Initialization
        :param max_depth, type: integer
        maximum depth of the regression tree. The maximum
        depth limits the number of nodes in the tree. Tune this parameter
        for best performance; the best value depends on the interaction
        of the input variables.
        :param min_samples_split, type: integer
        minimum number of samples required to split an internal node:

        root: type: dictionary, the root node of the regression tree.
        '''

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        '''
        Inputs:
        X: Train feature data, type: numpy array, shape: (N, num_feature)
        Y: Train label data, type: numpy array, shape: (N,)

        You should update the self.root in this function.
        '''
        
        self.root = self.tree_generator(X, y, self.max_depth, self.min_samples_split, 0)
        return self.root
    
    def tree_generator(self, X, y, maxdepth, minsamplessplit, depth):

        if depth==maxdepth or y.size<minsamplessplit:
            return y.mean()
        else:
            if (y==y.mean()).all():
                return y.mean()
            else:
                s, j = self.split_choose(X, y)
                X_l, X_r, y_l, y_r = self.split(X, y, s, j)
                # print()
                left_child = self.tree_generator(X_l, y_l, maxdepth, minsamplessplit, depth+1)
                right_child = self.tree_generator(X_r, y_r, maxdepth, minsamplessplit, depth+1)

                return {"splitting_variable": j, "splitting_threshold": s, "left": left_child, "right": right_child}
            
    def split_choose(self, X, y):
        n_feature = np.size(X,axis=1)
        error = float('inf')
        for j in range(0, n_feature):
            # print(X.shape)
            for s in (X[:,j]):
                indictor = X[:,j]<=s
                index_l = np.nonzero(indictor)
                index_r = np.nonzero(~indictor)
                y_1 = y[index_l]
                y_2 = y[index_r]
                if len(y_1)>0 and len(y_2)>0:
                    error_step = np.power(y_1-y_1.mean(),2).sum()+np.power(y_2-y_2.mean(),2).sum()
                    # if(j==9):
                    #     print('---------\n',y_1,'\n',y_1.mean(),'\n',y_2,'\n',y_2.mean())
                    #     print(error_step, j, s, index_l, index_r)
                        
                    if error_step<error:
                        error = error_step
                        j_opt = j
                        s_opt = s
        return s_opt, j_opt
                




    def split(self, X, y, s, j):
        indictor = X[:,j]<=s
        index_l = np.nonzero(indictor)
        index_r = np.nonzero(~indictor)

        # print("left X, right X: ", X[index_l].shape, X[index_r].shape)
        return X[index_l], X[index_r], y[index_l], y[index_r]

    def predict(self, X):
        '''
        :param X: Feature data, type: numpy array, shape: (N, num_feature)
        :return: y_pred: Predicted label, type: numpy array, shape: (N,)
        '''

        pred = []
        for i in X:
            pred.append(self.asktree(self.root, i))
        pred = np.array(pred)
        return pred
    
    def asktree(self, tree, X):
        if isinstance(tree, float):
            return tree
        else:
            index_feature = tree["splitting_variable"]
            threshold = tree["splitting_threshold"]
            if X[index_feature]<=threshold:
                return self.asktree(tree["left"],X)
            else:
                return self.asktree(tree["right"],X)
        

    def get_model_dict(self):
        model_dict = self.root
        return model_dict

    def save_model_to_json(self, file_name):
        model_dict = self.root
        with open(file_name, 'w') as fp:
            json.dump(model_dict, fp)


def compare_json_dic(json_dic, sample_json_dic):
    if isinstance(json_dic, dict):
        result = 1
        for key in sample_json_dic:
            if key in json_dic:
                result = result * compare_json_dic(json_dic[key], sample_json_dic[key])
                if result == 0:
                    return 0
            else:
                return 0
        return result
    else:
        rel_error = abs(json_dic - sample_json_dic) / np.maximum(1e-8, abs(sample_json_dic))
        if rel_error <= 1e-5:
            return 1
        else:
            return 0


def compare_predict_output(output, sample_output):
    rel_error = (abs(output - sample_output) / np.maximum(1e-8, abs(sample_output))).mean()
    if rel_error <= 1e-5:
        return 1
    else:
        return 0

# For test
if __name__=='__main__':
    for i in range(1):
        x_train = np.genfromtxt("Test_data" + os.sep + "x_" + str(i) +".csv", delimiter=",")
        y_train = np.genfromtxt("Test_data" + os.sep + "y_" + str(i) +".csv", delimiter=",")

        for j in range(2):
            tree = MyDecisionTreeRegressor(max_depth=5, min_samples_split=j + 2)
            tree.fit(x_train, y_train)

            model_dict = tree.get_model_dict()
            y_pred = tree.predict(x_train)

            with open("Test_data" + os.sep + "decision_tree_" + str(i) + "_" + str(j) + ".json", 'r') as fp:
                test_model_dict = json.load(fp)

            y_test_pred = np.genfromtxt("Test_data" + os.sep + "y_pred_decision_tree_"  + str(i) + "_" + str(j) + ".csv", delimiter=",")
            if compare_json_dic(model_dict, test_model_dict) * compare_predict_output(y_pred, y_test_pred) == 1:
                print("True")
            else:
                print("False")



