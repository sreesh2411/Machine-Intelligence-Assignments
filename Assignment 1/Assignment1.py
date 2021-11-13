'''
Assume df is a pandas dataframe object of the dataset given
'''
import numpy as np
import pandas as pd
import random

eps = np.finfo(float).eps #since log 0 and 0 in denominator is not possible, we are using eps to avoid error.

'''Calculate the entropy of the enitre dataset'''
	#input:pandas_dataframe
	#output:int/float/double/large

def get_entropy_of_dataset(df):
    entropy = 0
    target = df.iloc[:,-1]
    t_values = target.unique()
    for x in t_values:
        #print(target.value_counts())
        fraction = (target.value_counts()[x])/len(target)
        entropy += -fraction*np.log2(fraction)
    #print(entropy)
    return entropy



'''Return entropy of the attribute provided as parameter'''
	#input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
	#output:int/float/double/large
def get_entropy_of_attribute(df,attribute):
    target = df.iloc[:,-1]
    t_values = target.unique()
    att_values = df[attribute].unique()

    entropy_of_attribute = 0
    for x in att_values:
        entropy_attfeatures = 0
        for y in t_values:
            numerator = len(df[attribute][df[attribute] == x][target == y])
            denominator = len(df[attribute][df[attribute] == x])
            fraction = numerator/(denominator + eps)
            entropy_attfeatures +=  -fraction*np.log2(fraction + eps)
        fraction2 = denominator/len(df)
        entropy_of_attribute += -fraction2*entropy_attfeatures

    #print(entropy_of_attribute)
    return abs(entropy_of_attribute)



'''Return Information Gain of the attribute provided as parameter'''
	#input:int/float/double/large,int/float/double/large
	#output:int/float/double/large
def get_information_gain(df,attribute):
    information_gain = 0
    #print(information_gain)
    information_gain = get_entropy_of_dataset(df) - get_entropy_of_attribute(df,attribute)
    #print(information_gain)
    return information_gain



''' Returns Attribute with highest info gain'''  
	#input: pandas_dataframe
	#output: ({dict},'str')     

def get_selected_attribute(df):
   
    information_gains={}
    selected_column=''
    
	
    attr_entropy = {feature : get_entropy_of_attribute(df,feature) for feature in df.keys()}
    #print(attr_entropy)
    for x in attr_entropy:
        #print(x)
        #print(attr_entropy[x])
        information_gains.update({x : get_information_gain(df,x)})
	 	

    selected_column = str(df.keys()[:-1][np.argmax(information_gains)])
    #print(information_gains)

    '''
	Return a tuple with the first element as a dictionary which has IG of all columns 
	and the second element as a string with the name of the column selected

	example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
	'''

    return (information_gains,selected_column)


'''
------- TEST CASES --------
How to run sample test cases ?

Simply run the file DT_SampleTestCase.py
Follow convention and do not change any file / function names

'''