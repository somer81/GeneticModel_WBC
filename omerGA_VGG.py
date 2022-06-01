# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:46:09 2020

@author: AHC
"""


import numpy as np
import tensorflow as tf
from keras.models import Sequential
#from keras.dataset import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import beckend as K
from keras.optimizers import SGD
#from keras.utils import np_utils
from tensorflow.keras import utils


from keras.utils import to_categorical
from keras import optimizers
from tqdm import tqdm
import os
import cv2
import scipy
from PIL import Image
from skimage.transform import resize

################################################
### CALCULATING THE OBJECTIVE FUNCTION VALUE ###
################################################

def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    #z = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 1
                label2 = 1
            elif wbc_type in ['EOSINOPHIL']:
                label = 2
                label2 = 1
            elif wbc_type in ['MONOCYTE']:
                label = 3
                label2 = 0
            elif wbc_type in ['LYMPHOCYTE']:
                label = 4
                label2 = 0
            else:
                label = 5
                label2 = 0
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img_file = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img_file is not None:
                    #img_file = img_file.resize((60, 80,3),Image.ANTIALIAS)
                    #img_file = resize(img_file, output_shape=(60, 80)).reshape((1, 60 * 80 * 3)).T
                    #np.array(Image.fromarray(img_file).resize(60, 80),PIL.Image.BILINEAR).astype(np.double)
                    img_file = scipy.misc.imresize(arr=img_file, size=(60, 80, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
                    #z.append(label2)
    X = np.asarray(X)
    y = np.asarray(y)
    #z = np.asarray(z)
    #return X,y,z
    return X,y



#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

x_train, y_train = get_data('C:/Doktora_Tez/dataset2-master/images/TRAIN/')
x_test, y_test = get_data('C:/Doktora_Tez/dataset2-master/images/TEST/')

# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
y_trainHot = to_categorical(y_train, num_classes = 5)
y_testHot = to_categorical(y_test, num_classes = 5)


#y_trainHot = to_categorical(y_train, num_classes = 10)
#y_testHot = to_categorical(y_test, num_classes = 10)

#dict_characters = {1:'NEUTROPHIL',2:'EOSINOPHIL',3:'MONOCYTE',4:'LYMPHOCYTE'}
#dict_characters2 = {0:'Mononuclear',1:'Polynuclear'}
#print(dict_characters)
#print(dict_characters2)





# calculate fitness value for the chromosome of 0s and 1s
def objective_value(chromosome):  
    
    lb_x = 1 # lower bound for chromosome x
    ub_x = 63 # upper bound for chromosome x
    len_x = 6 # length of chromosome x // Number of Filters 
    lb_y = 32 # lower bound for chromosome y
    ub_y = 128 # upper bound for chromosome y
    len_y = 7 # length of chromosome y  // Batch size 
    lb_z = 1 # lower bound for chromosome x
    ub_z = 7 # upper bound for chromosome x
    len_z = 3 # length of chromosome x  // filter size 
    lb_t = 10 # lower bound for chromosome y
    ub_t = 30 # upper bound for chromosome y
    len_t = 5 # length of chromosome y // dropout 
    
    precision_x = (ub_x-lb_x)/((2**len_x)-1) # precision for decoding x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # precision for decoding y
    precision_z = (ub_z-lb_z)/((2**len_z)-1) # precision for decoding x
    precision_t = (ub_t-lb_t)/((2**len_t)-1) # precision for decoding y
    
    z = 0 # because we start at 2^0, in the formula
    i = 1 # because we start at the very last element of the vector [index -1]
    t_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len_t):
        t_bit = chromosome[-i]*(2**z)
        t_bit_sum = t_bit_sum + t_bit
        i = i+1
        z = z+1 
        
    z = 0 # because we start at 2^0, in the formula
    i = 1 + len_t # because we start at the very last element of the vector [index -1]
    z_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len_z):
        z_bit = chromosome[-i]*(2**z)
        z_bit_sum = z_bit_sum + z_bit
        i = i+1
        z = z+1 
        
    z = 0 # because we start at 2^0, in the formula
    i = 1 + len_z + len_t # because we start at the very last element of the vector [index -1]
    y_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len_y):
        y_bit = chromosome[-i]*(2**z)
        y_bit_sum = y_bit_sum + y_bit
        i = i+1
        z = z+1 
    
    z = 0 # because we start at 2^0, in the formula
    t = 1 + len_y + len_z + len_t # [6,8,3,9] (first 2 are y, so index will be 1+2 = -3)
    x_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for j in range(len_x):
        x_bit = chromosome[-t]*(2**z)
        x_bit_sum = x_bit_sum + x_bit
        t = t+1
        z = z+1
    
    # the formulas to decode the chromosome of 0s and 1s to an actual number, the value of x or y
    decoded_x = (x_bit_sum*precision_x)+lb_x
    decoded_y = (y_bit_sum*precision_y)+lb_y
    decoded_z = (z_bit_sum*precision_z)+lb_z
    decoded_t = (t_bit_sum*precision_t)+lb_t
    
    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
    x_train, y_train = get_data('C:/Doktora_Tez/dataset2-master/images/TRAIN/')
    x_test, y_test = get_data('C:/Doktora_Tez/dataset2-master/images/TEST/')
    

    img_rows = int(x_train[0].shape[0])
    img_cols = int(x_train[1].shape[1])
    
    x_train = x_train.reshape(int(x_train.shape[0]), img_rows, img_cols,3)
    x_test  = x_test.reshape(int(x_test.shape[0]), img_rows, img_cols,3)
    
    input_shape = (int(img_rows), int(img_cols), 3)
    
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    
    x_train /= 255
    x_test /= 255
    decoded_t /= 100
    
    y_train = utils.to_categorical(y_train)
    y_test  = utils.to_categorical(y_test)
    
    num_classes = int(y_test.shape[1])
    #num_pixels = x_train.shape[1] * x_train.shape[2]
    
    model = Sequential()
    model.add(Conv2D(int(decoded_x), kernel_size=(int(decoded_z),int(decoded_z)), 
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(int(decoded_x), (int(decoded_z),int(decoded_z)), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(decoded_t))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(decoded_t))
    model.add(Dense(num_classes,activation='softmax'))
    
    model.compile(loss= 'categorical_crossentropy',
                  #optimizer = SGD(0.01),
                  optimizer = 'Adam',
                  metrics= ['accuracy'])
    
    batch_size = decoded_y
    
    epochs = 5
    
    history = model.fit(x_train,
                        y_train,
                        int(batch_size),
                        int(epochs),
                        verbose =1,
                        validation_data = (x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    
    #obj_function_value = score[1]
    
    
    # the himmelblau function
    # min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    # objective function value for the decoded x and decoded y
    obj_function_value = ((decoded_x**2)+decoded_y-11)**2+(decoded_z+(decoded_t**2)-7)**2
    
    return decoded_x,decoded_y,obj_function_value # the defined function will return 3 values



#################################################
### SELECTING TWO PARENTS FROM THE POPULATION ###
### USING TOURNAMENT SELECTION METHOD ###########
#################################################

# finding 2 parents from the pool of solutions
# using the tournament selection method 
def find_parents_ts(all_solutions):
    
    # make an empty array to place the selected parents
    parents = np.empty((0,np.size(all_solutions,1)))
    
    for i in range(2): # do the process twice to get 2 parents
        
        # select 3 random parents from the pool of solutions you have
        
        # get 3 random integers
        indices_list = np.random.choice(len(all_solutions),3,replace=False)
        
        # get the 3 possible parents for selection
        posb_parent_1 = all_solutions[indices_list[0]]
        posb_parent_2 = all_solutions[indices_list[1]]
        posb_parent_3 = all_solutions[indices_list[2]]
        
        # get objective function value (fitness) for each possible parent
        # index no.2 because the objective_value function gives the fitness value at index no.2
        obj_func_parent_1 = objective_value(posb_parent_1)[2] # possible parent 1
        obj_func_parent_2 = objective_value(posb_parent_2)[2] # possible parent 2
        obj_func_parent_3 = objective_value(posb_parent_3)[2] # possible parent 3
        
        # find which parent is the best
        min_obj_func = max(obj_func_parent_1,obj_func_parent_2,obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        # put the selected parent in the empty array we created above
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:] # parent_1, first element in the array
    parent_2 = parents[1,:] # parent_2, second element in the array
    
    return parent_1,parent_2 # the defined function will return 2 arrays



####################################################
### CROSSOVER THE TWO PARENTS TO CREATE CHILDREN ###
####################################################

# crossover between the 2 parents to create 2 children
# functions inputs are parent_1, parent_2, and the probability you would like for crossover
# default probability of crossover is 1
def crossover(parent_1,parent_2,prob_crsvr=1):
    
    child_1 = np.empty((0,len(parent_1)))
    child_2 = np.empty((0,len(parent_2)))
    
    
    rand_num_to_crsvr_or_not = np.random.rand() # do we crossover or no???
    
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(0,len(parent_2))
        
        # get different indices
        # to make sure you will crossover at least one gene
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_1))
        
        
        # IF THE INDEX FROM PARENT_1 COMES BEFORE PARENT_2
        # e.g. parent_1 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        # e.g. parent_2 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        if index_1 < index_2:
            
            
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            
            # first_seg_parent_1 -->
            # for parent_1: the genes from the beginning of parent_1 to the
                    # beginning of the middle segment of parent_1
            first_seg_parent_1 = parent_1[:index_1]
            
            # middle segment; where the crossover will happen
            # for parent_1: the genes from the index chosen for parent_1 to
                    # the index chosen for parent_2
            mid_seg_parent_1 = parent_1[index_1:index_2+1]
            
            # last_seg_parent_1 -->
            # for parent_1: the genes from the end of the middle segment of
                    # parent_1 to the last gene of parent_1
            last_seg_parent_1 = parent_1[index_2+1:]
            
            
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            
            # first_seg_parent_2 --> same as parent_1
            first_seg_parent_2 = parent_2[:index_1]
            
            # mid_seg_parent_2 --> same as parent_1
            mid_seg_parent_2 = parent_2[index_1:index_2+1]
            
            # last_seg_parent_2 --> same as parent_1
            last_seg_parent_2 = parent_2[index_2+1:]
            
            
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            
            # the first segmant from parent_1
            # plus the middle segment from parent_2
            # plus the last segment from parent_1
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            
            
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            
            # the first segmant from parent_2
            # plus the middle segment from parent_1
            # plus the last segment from parent_2
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
        
        
        
        ### THE SAME PROCESS BUT INDEX FROM PARENT_2 COMES BEFORE PARENT_1
        # e.g. parent_1 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        # e.g. parent_2 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        else:
            
            first_seg_parent_1 = parent_1[:index_2]
            mid_seg_parent_1 = parent_1[index_2:index_1+1]
            last_seg_parent_1 = parent_1[index_1+1:]
            
            first_seg_parent_2 = parent_2[:index_2]
            mid_seg_parent_2 = parent_2[index_2:index_1+1]
            last_seg_parent_2 = parent_2[index_1+1:]
            
            
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
     
    # when we will not crossover
    # when rand_num_to_crsvr_or_not is NOT less (is greater) than prob_crsvr
    # when prob_crsvr == 1, then rand_num_to_crsvr_or_not will always be less
            # than prob_crsvr, so we will always crossover then
    else:
        child_1 = parent_1
        child_2 = parent_2
    
    return child_1,child_2 # the defined function will return 2 arrays



############################################################
### MUTATING THE TWO CHILDREN TO CREATE MUTATED CHILDREN ###
############################################################

# mutation for the 2 children
# functions inputs are child_1, child_2, and the probability you would like for mutation
# default probability of mutation is 0.2
def mutation(child_1,child_2,prob_mutation=0.2):
    
    # mutated_child_1
    mutated_child_1 = np.empty((0,len(child_1)))
      
    t = 0 # start at the very first index of child_1
    for i in child_1: # for each gene (index)
        
        rand_num_to_mutate_or_not_1 = np.random.rand() # do we mutate or no???
        
        # if the rand_num_to_mutate_or_not_1 is less that the probability of mutation
                # then we mutate at that given gene (index we are currently at)
        if rand_num_to_mutate_or_not_1 < prob_mutation:
            
            if child_1[t] == 0: # if we mutate, a 0 becomes a 1
                child_1[t] = 1
            
            else:
                child_1[t] = 0  # if we mutate, a 1 becomes a 0
            
            mutated_child_1 = child_1
            
            t = t+1
        
        else:
            mutated_child_1 = child_1
            
            t = t+1
    
       
    # mutated_child_2
    # same process as mutated_child_1
    mutated_child_2 = np.empty((0,len(child_2)))
    
    t = 0
    for i in child_2:
        
        rand_num_to_mutate_or_not_2 = np.random.rand() # prob. to mutate
        
        if rand_num_to_mutate_or_not_2 < prob_mutation:
            
            if child_2[t] == 0:
                child_2[t] = 1
           
            else:
                child_2[t] = 0
            
            mutated_child_2 = child_2
            
            t = t+1
        
        else:
            mutated_child_2 = child_2
            
            t = t+1
    
    return mutated_child_1,mutated_child_2 # the defined function will return 2 arrays




