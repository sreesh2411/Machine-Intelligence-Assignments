'''
Design of a Neural Network from scratch

*************<IMP>*************
Mention hyperparameters used and describe functionality in detail in this space
- carries 1 mark
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class NN:
    '''
    def __init__(self, weights_hid, bias):
        self.weights_hid = weights_hid
        self.bias = bias
    '''    
    ''' X and Y are dataframes '''
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def sigmoid_derivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
        
    def relu(self,z):
        return np.maximum(0, z)
        
    def relu_prime(self,z):
        return np.where(z > 0, 1.0, 0.0)
        
    def softmax(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def fit(self,X,Y):
        '''
        Function that trains the neural network by taking x_train and y_train samples as input
        '''
        global weights_hid
        global bias_hid
        global weights_out
        global bias_out
        input_set = np.array(X)
        #print(input_set)
        labels = np.array(Y)
        labels = labels.reshape(len(Y),1)
        #one_hot_labels = np.zeros((len(Y),2))
        #print(one_hot_labels)
        #for i in range(len(Y)):
        #    one_hot_labels[i, int(labels[i])] = 1
        #np.random.seed(0)
        lr = 0.09 #learning rate
        for epoch in range(40000):
            inputs = input_set
            XW = np.dot(inputs, weights_hid) + bias_hid
            z = self.sigmoid(XW)
            
            XW_out = np.dot(z, weights_out) + bias_out
            z2 = self.sigmoid(XW_out)
            '''
            error_out = (np.sum((z2 - labels)**2) / labels.size) * self.sigmoid_derivative(XW_out)
            error_hid = np.dot(error_out , weights_out.T) * self.sigmoid_derivative(XW)
            dW_out = np.dot(z.T , error_out)
            dW_hid = np.dot(inputs.T,error_hid)
            weights_hid = weights_hid - (lr * dW_hid)
            weights_out = weights_out - (lr * dW_out)
            '''
            
            error_out = ((1 / 2) * (np.power((z2 - labels), 2)))
            #print(error_out.sum())

            dcost_dz2 = z2 - labels
            dz2_dxwo = self.sigmoid_derivative(XW_out) 
            #dcost_dxwo = z2 - one_hot_labels
            dxwo_dwo = z

            dcost_wo = np.dot(dxwo_dwo.T, dcost_dz2 * dz2_dxwo)
            #dcost_wo = np.dot(dxwo_dwo.T, dcost_dxwo)
            #dcost_bo = dcost_dxwo

    # Phase 2 =======================

    # dcost_w1 = dcost_dah * dah_dzh * dzh_dw1
    # dcost_dah = dcost_dzo * dzo_dah
            dcost_dxwo = dcost_dz2 * dz2_dxwo
            dxwo_dz = weights_out
            dcost_dz = np.dot(dcost_dxwo , dxwo_dz.T)
            dz_dxw = self.sigmoid_derivative(XW) 
            dxw_dwh = inputs
            dcost_wh = np.dot(dxw_dwh.T, dz_dxw * dcost_dz)
            
            #dcost_bh = dcost_dz * dz_dxw

    # Update Weights ================

            weights_hid -= lr * dcost_wh
            weights_out -= lr * dcost_wo
            
            #bias_hid -= lr * dcost_bh.sum(axis=0)
            #bias_out -= lr * dcost_bo.sum(axis=0)
            
        #print(weights_hid)
	
    def predict(self,X):

        """
		The predict function performs a simple feed forward of weights
		and outputs yhat values 

		yhat is a list of the predicted value for df X
		"""
        global weights_hid
        global bias_hid
        global weights_out
        global bias_out
        #print(weights_hid)
        single_pt = np.array(X)
        yhat = self.sigmoid(np.dot(single_pt, weights_hid) + bias_hid)
        yhat2 = self.sigmoid(np.dot(yhat, weights_out) + bias_out)
        #print(yhat)
        #print(yhat2)
        #yhat2 = np.argmax(yhat2,axis=1)
        #print(yhat2)
        return yhat2

    def CM(y_test,y_test_obs):
        '''
		Prints confusion matrix 
		y_test is list of y values in the test dataset
		y_test_obs is list of y values predicted by the model

		'''

        for i in range(len(y_test_obs)):
            if(y_test_obs[i]>0.6):
                y_test_obs[i]=1
            else:
                y_test_obs[i]=0
		
        #print(y_test_obs,y_test)
        cm=[[0,0],[0,0]]
        fp=0
        fn=0
        tp=0
        tn=0
		
        for i in range(len(y_test)):
            if(y_test[i]==1 and y_test_obs[i]==1):
                tp=tp+1
            if(y_test[i]==0 and y_test_obs[i]==0):
                tn=tn+1
            if(y_test[i]==1 and y_test_obs[i]==0):
                fp=fp+1
            if(y_test[i]==0 and y_test_obs[i]==1):
                fn=fn+1
        cm[0][0]=tn
        cm[0][1]=fp
        cm[1][0]=fn
        cm[1][1]=tp

        p= tp/(tp+fp)
        a= (tp+tn)/(tp+tn+fp+fn)
        r=tp/(tp+fn)
        f1=(2*p*r)/(p+r)
		
        print("Confusion Matrix : ")
        print(cm)
        print("\n")
        print(f"Accuracy: {a}")
        print(f"Precision : {p}")
        print(f"Recall : {r}")
        print(f"F1 SCORE : {f1}")
        
def normalize(dataset):
        dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))
        dataNorm["Result"]=dataset["Result"]
        dataNorm["IFA"]=dataset["IFA"]
        return dataNorm

if __name__=="__main__":
	df = pd.read_csv('LBW_Dataset.csv')

	df['HB'].fillna(value=df['HB'].mean(), inplace=True)
	df['BP'].fillna(value=df['BP'].mean(), inplace=True)
	df['Education'].fillna(value=df['Education'].mean(), inplace=True)
	df['Weight'].fillna(value=int(df['Weight'].mean()), inplace=True)
	df['Age'].fillna(value=int(df['Age'].mean()), inplace=True)
	df['Residence'].fillna(value=df['Residence'].mode()[0], inplace=True)
	df['Delivery phase'].fillna(value=df['Delivery phase'].mode()[0], inplace=True)

	df = df.drop(['Education', 'Delivery phase'], axis = 1)
	'''
	df = preprocessing.normalize(df,norm='l1')
	df = pd.DataFrame(df)
	'''
	'''
	x = df.values #returns a numpy array
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	df = pd.DataFrame(x_scaled)
	'''
	df=normalize(df)
	#print(df)
	
	X = df.iloc[:, 0:7].values
	Y = df.iloc[:, 7].values
	    
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state = 2)

	#x_train = preprocessing.normalize(x_train,norm='l1')
	#x_train = pd.DataFrame(x_train)
	#x_test = preprocessing.normalize(x_test,norm='l1')
	#x_test = pd.DataFrame(x_test)
	
	
	input_size = 7
	hidden_size = 8
	output_size = 1

	#np.random.seed(42)
	weights_hid = np.random.rand(input_size, hidden_size) * np.sqrt(2.0/input_size)
	#print(weights_hid)
	bias_hid= np.full((1, hidden_size), np.random.rand(1))
	
	weights_out = np.random.rand(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
	bias_out = np.full((1, output_size), np.random.rand(1))
	
	nn = NN()
	nn.fit(x_train, y_train)
	
	print("\n\n\t\tTesting Accuracy:")
	y_test_obs = nn.predict(x_test)
	NN.CM(y_test, y_test_obs)
	
	print("\n\n\t\tTraining Accuracy:")
	y_train_obs = nn.predict(x_train)
	NN.CM(y_train, y_train_obs)
	#print(weights_hid)

