#!/usr/bin/env python
# coding: utf-8

# In[57]:


import matplotlib
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import pywt
import tensorflow as tf
import pickle
import random
import re
import os
from PIL import Image
import time
import math


# In[58]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# **Create data set with circles in random positions**

# In[62]:


#Number of points on circle is drawn from Unif([m-d,m+d])
m = 20
d = 5

seed(1)

def circle(center,radius):
    #print(center, radius)
    num_pts = random.randint(m-d, m+d)
    #print("This circle has {} points".format(num_pts))
    thetas = [random.random()*2*math.pi for i in range(num_pts)]
    #print("num_pts = {}".format(num_pts))
    #print("thetas = ", thetas)
    circ = np.vstack([radius*np.array([np.cos(theta),np.sin(theta)]) + center for theta in thetas])
    return circ

#truncate where it's zero, then round to next highest power of 2
def get_trunc_ind(arr):
    for i,_ in enumerate(arr):
        if np.all(arr[i:] == np.zeros(len(arr[i:]))):
            return int(2**(np.ceil(np.log(i)/np.log(2))))
    return len(arr)

def get_len_and_pos(ind,arr):
    #assume ind > 0
    n = np.log(len(arr))/np.log(2)
    wavelength = 2**(n-int((np.log(ind)/np.log(2))))
    pos = (ind-2**(int(np.log(ind)/np.log(2))))*wavelength
    return int(wavelength),int(pos)

def get_output(max_data,all_coeffs):
    coeff,ind = max_data
    if ind == 0:
        return coeff/np.sqrt(len(all_coeffs)),len(all_coeffs),0
    else:
        wl, pos = get_len_and_pos(ind,all_coeffs)
        return coeff/np.sqrt(wl),wl,pos
    
def convert_outputs(x):
    if x == 0:
        return 0
    else:
        return np.log(x)/np.log(2)
    
def pad_and_add(x,y):
    if len(x) >= len(y):
        y = np.pad(y,(0,len(x)-len(y)),'constant',constant_values=0)
    else:
        x = np.pad(x,(0,len(y)-len(x)),'constant',constant_values=0)
    return x + y

def plot_outputs(output1,output2,output3):
    func = np.zeros(1)
    out2 = np.round(output2)
    out3 = np.round(output3)
    for i,coeff in enumerate(output1[:-1]):
        wl = 2**int(np.round(output2[i]))
        
        temppos = int(np.round(output3[i]))
        if temppos == 0:
            pos = 0
        else:
            pos = 2**temppos
        
        to_add = np.hstack([np.zeros(pos),coeff*np.ones(wl//2),(-1)*coeff*np.ones(wl//2)])
        func = pad_and_add(func,to_add)
    func += output1[-1]
    return func
    


# In[64]:


#test = circle((0.5,0.5), 0.5)
#plt.scatter(test[:,0],test[:,1])


# In[65]:


#Take the N largest absolute values of coefficients. 
N = 20

#Max number of circles. Each sample will have K circles, where K is drawn from Unif([0,max_num_circles])
max_num_circles = 10

#Number of samples to generate
M = 5

try:
    with open('outputs_bigger.p','rb') as f:
        outputs = pickle.load(f)
    
    with open('inputs_bigger.p','rb') as f:
        inputs = pickle.load(f)
    
except Exception as e:
    t0 = time.clock()
    inputs = []
    outputs = []
    for i in range(0,M):
        num_circles = random.randint(1, max_num_circles)
        centers = np.random.random(size=(num_circles,2))
        radii = np.random.random(size=(num_circles,1))*0.5
        circle_pts = np.vstack([circle(center,radius) for (center,radius) in zip(centers,radii)])
        inputs.append(circle_pts)
        rc = gd.RipsComplex(circle_pts)
        st = rc.create_simplex_tree(max_dimension=2)
        st.persistence()
        temp = st.get_filtration()
        filtrants = sorted(set({x for _,x in temp}))

        betti_data = np.array([st.persistent_betti_numbers(start,end) for start,end in zip(filtrants[:-1],filtrants[1:])])
        trunced_arr = betti_data[:get_trunc_ind(betti_data[:,1]),1]

        coeffs = pywt.wavedec(trunced_arr,wavelet='db1',level=None)
        all_coeffs = np.hstack(coeffs)
        max_inds = np.argpartition(np.abs(all_coeffs),-N)[-N:]
        sorted_max_data = [(x,y) for x,y in sorted(list(zip(all_coeffs[max_inds],max_inds)))]
        output = np.array([get_output(x,all_coeffs) for x in sorted_max_data])
        outputs.append(output)
        if ((i+1)%100) == 0:
            print("Generated {} samples in {} seconds".format(i+1, time.clock() - t0))
    print("Generated {} samples in {} seconds".format(i+1, time.clock() - t0))


# In[47]:


with open('outputs_bigger.p','wb') as f:
    pickle.dump(outputs,f)
    
with open('inputs_bigger.p','wb') as f:
    pickle.dump(inputs,f)


# In[10]:


arr_inputs = np.array(inputs)


# In[11]:


outputs1 = np.array([x[:,0] for x in outputs])
outputs2 = np.array([x[:,1] for x in outputs])
outputs3 = np.array([x[:,2] for x in outputs])


# In[12]:


outputs2log = np.array([[convert_outputs(x) for x in arr] for arr in outputs2])
outputs3log = np.array([[convert_outputs(x) for x in arr] for arr in outputs3])


# ### Make images from inputs

# In[245]:




def conv_layer(Input,activation='relu',outname='conv'):
    Conv1 = tf.keras.layers.Conv1D(64,kernel_size=3,strides=1,padding='same',input_shape=(300,2),activation=activation)(Input)
    Pooling1 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2)(Conv1)
    Conv2 = tf.keras.layers.Conv1D(64,kernel_size=3,strides=1,padding='same',activation=activation)(Pooling1)
    Pooling2 = tf.keras.layers.MaxPooling1D(pool_size=2,strides=2)(Conv2)
    Flat = tf.keras.layers.Flatten()(Pooling2)
    Dense1 = tf.keras.layers.Dense(32,activation=activation)(Flat)
    Out = tf.keras.layers.Dense(10,activation=activation,name=outname)(Dense1)
    return Out
    
def res_block_coeff(Input,activation='linear',bias_reg = None,outname='coeff'):
    Start = tf.keras.layers.Flatten()(Input)
    Dense1 = tf.keras.layers.Dense(100, activation=activation)(Start)
    #BN1 = tf.compat.v2.keras.layers.BatchNormalization()(Dense1)
    Dropout1 = tf.keras.layers.Dropout(0.2)(Dense1)
    Dense2 = tf.keras.layers.Dense(70,activation=activation)(Dropout1)
    #BN2 = tf.compat.v2.keras.layers.BatchNormalization()(Dense2)
    Dense3 = tf.keras.layers.Dense(40,activation=activation)(Dense2)
    #BN3 = tf.compat.v2.keras.layers.BatchNormalization()(Dense3)
    #Dense4 = tf.keras.layers.Dense(20,activation=activation)(BN3)
    to_add = tf.keras.layers.Dense(40,activation=activation,kernel_initializer = tf.keras.initializers.zeros())(Start)
    Add = tf.keras.layers.add([to_add,Dense3])
    Out = tf.keras.layers.Dense(10,activation = activation,bias_regularizer = bias_reg,name=outname)(Add)
    return Out

def res_block_wl(Input,activation='relu',bias_reg = None,outname='Dense'):
    Start = tf.keras.layers.Flatten()(Input)
    Dense1 = tf.keras.layers.Dense(100, activation=activation,
                                                 kernel_regularizer = tf.keras.regularizers.l2(0.01))(Start)
    BN1 = tf.compat.v2.keras.layers.BatchNormalization()(Dense1)
    Dropout1 = tf.keras.layers.Dropout(0.2)(BN1)
    Dense2 = tf.keras.layers.Dense(70,activation=activation)(Dropout1)
    #BN2 = tf.compat.v2.keras.layers.BatchNormalization()(Dense2)
    Dense3 = tf.keras.layers.Dense(40,activation=activation)(Dense2)
    #BN3 = tf.compat.v2.keras.layers.BatchNormalization()(Dense3)
    #Dense4 = tf.keras.layers.Dense(20,activation=activation)(BN3)
    #to_add = tf.keras.layers.Dense(40,activation=activation,kernel_initializer = tf.keras.initializers.Constant(1/300))(Start)
    #Add = tf.keras.layers.add([to_add,Dense3])
    Out = tf.keras.layers.Dense(10,activation = activation,bias_regularizer = bias_reg,name=outname)(Dense3)
    return Out


# In[246]:


Input = tf.keras.layers.Input((300,2))

Out1 = res_block_coeff(Input,activation='linear')
Out2 = res_block_wl(Input,activation='relu',bias_reg = tf.keras.regularizers.l2(0.01),outname='wl')
Out3 = res_block_wl(Input,activation='relu',bias_reg = tf.keras.regularizers.l2(0.01),outname='pos')

model = tf.keras.Model(inputs = Input, outputs = [Out1,Out2,Out3])
#tf.keras.utils.plot_model(model,to_file='model.png')
model.compile(optimizer='adam',
              loss='mean_squared_error',
             )#metrics=['accuracy'])


# In[247]:


Input_conv = tf.keras.layers.Input((300,2))

Out1_conv = conv_layer(Input_conv,activation='linear',outname='coeff')
Out2_conv = conv_layer(Input_conv,activation='relu',outname='wl')
Out3_conv = conv_layer(Input_conv,activation='relu',outname='pos')

model_conv = tf.keras.Model(inputs = Input_conv, outputs = [Out1_conv,Out2_conv,Out3_conv])
model_conv.compile(optimizer='adam',
              loss='mean_squared_error',
             )


# In[248]:


logdir="logs/fit/"
get_ipython().system('rm -r ./logs/fit')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# ### Train the vanilla NN

# In[249]:


model.fit(arr_inputs,[outputs1,outputs2log,outputs3log],epochs=100,callbacks=[tensorboard_callback])


# In[319]:


#%tensorboard --logdir logs


# In[224]:


model.predict(arr_inputs[0:1,:,:])


# In[13]:


for i in range(0,5):
    plt.figure()
    plt.plot(plot_outputs(outputs1[i],outputs2log[i],outputs3log[i]))
    sample_input = arr_inputs[i:i+1][0].copy()
    #random.shuffle(sample_input)
    sample_input = np.expand_dims(sample_input,0)
    plt.plot(plot_outputs(*[x[0] for x in model.predict(sample_input)]))


# ### Train the CNN

# In[252]:


model_conv.fit(arr_inputs,[outputs1,outputs2log,outputs3log],epochs=100,callbacks=[tensorboard_callback])


# In[14]:


for i in range(0,5):
    plt.figure()
    plt.plot(plot_outputs(outputs1[i],outputs2log[i],outputs3log[i]))
    sample_input = arr_inputs[i:i+1][0].copy()
    #random.shuffle(sample_input)
    sample_input = np.expand_dims(sample_input,0)
    plt.plot(plot_outputs(*[x[0] for x in model_conv.predict(sample_input)]))


# ### Set up for conv2D

# In[48]:


plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
plt.xticks([])
plt.yticks([])
plt.scatter(inputs[0][:,0],inputs[0][:,1],color='black')


# In[262]:


plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
for i,inp in enumerate(inputs):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.scatter(inp[:,0],inp[:,1],color='black')
    plt.savefig('images/points'+str(i)+'.png',bbox_inches='tight')
    plt.close()


# In[271]:


def get_sort_key(filename):
    return int(re.findall(r'(\d+)',filename)[0])


# In[273]:


sorted_images = sorted(os.listdir('images'),key=get_sort_key)


# In[274]:


sorted_images[0]


# In[293]:


test = 1-np.array(Image.open('images/' + sorted_images[0]).convert('L'))/255


# In[294]:


test.shape


# In[296]:


im_data = np.array([1-np.array(Image.open('images/' + imname).convert('L'))/255 for imname in sorted_images])


# In[312]:


im_data = np.expand_dims(im_data,-1)
im_data.shape


# In[313]:


def conv2_layer(Input,activation='relu',outname='conv'):
    Start = tf.keras.layers.Convolution2D(64,kernel_size=(3,3),strides=(1,1),input_shape=(238,356,1),padding='same',activation=activation)(Input)
    Pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(Start)
    Conv2 = tf.keras.layers.Convolution2D(64,kernel_size=(3,3),strides=(1,1),padding='same',activation=activation)(Pool1)
    Pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(Conv2)
    Flat = tf.keras.layers.Flatten()(Pool2
    #N = number of coefficients to take
    Out = tf.keras.layers.Dense(N,activation=activation,name=outname)(Flat)
    return Out


# In[320]:


Input_conv2 = tf.keras.layers.Input((238,356,1))
Out1_conv2 = conv2_layer(Input_conv2,activation='linear',outname='coeffs')
Out2_conv2 = conv2_layer(Input_conv2,activation='relu',outname='wl')
Out3_conv2 = conv2_layer(Input_conv2,activation='relu',outname='pos')

model_conv2 = tf.keras.Model(inputs = Input_conv2, outputs = [Out1_conv2,Out2_conv2,Out3_conv2])
model_conv2.compile(optimizer='adam',
              loss='mean_squared_error',
             )


# In[323]:


model_conv2.fit(im_data,[outputs1,outputs2log,outputs3log],epochs=15,callbacks=[tensorboard_callback])


# In[324]:


for i in range(0,15):
    plt.figure()
    plt.plot(plot_outputs(outputs1[i],outputs2log[i],outputs3log[i]))
    sample_input = im_data[i:i+1][0].copy()
    #random.shuffle(sample_input)
    sample_input = np.expand_dims(sample_input,0)
    plt.plot(plot_outputs(*[x[0] for x in model_conv2.predict(sample_input)]))


# In[ ]:




