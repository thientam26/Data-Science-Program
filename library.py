#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import numpy
import scipy
from scipy.linalg import lu
import random


# In[13]:


def create_maxtric(m,n):
    lst = []
    for i in range(m):
        lst_sub = []
        for j in range(n):
            s = "M["+str(i+1)+","+str(j+1)+"]:"
            x = eval(input(s))
            lst_sub.append(x)
        lst.append(lst_sub)
    return np.array(lst)


# In[14]:


def create_vector(n):
    lst =[]
    for i in range(n):
        s = 'v[' + str(i+1) + ']:'
        x = eval(input(s))
        lst.append(lst_sub)
    return np.array(lst)


# In[18]:


def create_vector_random(n,start,end):
    lst = []
    for i in range(n):
        x =random.randint(start,end+1)
        lst.append(x)
    return np.array(lst)


# In[15]:


def create_matrix_random(m,n,start,end):
    lst = []
    for i in range(m):
        lst_sub = []
        for j in range(n):
            x = random.randint(start, end+1)
            lst_sub.append(x)
        lst.append(lst_sub)
    return np.array(lst)


# In[16]:


def cal_vectors(op,u,v):
    result = None
    if op == '+':
        result = u+v
    elif op =='-':
        result = u-v
    elif op == '/':
        result = u/v
    elif op == '*':
        result = u*v
    elif op == 'dot':
        result = u.dot(v)
    else:
        result = None
    return result


# In[17]:


def cal_matrices(op,m1,m2):
    result = None
    if op == "+":
        result = m1 + m2
    elif op == "-":
        result = m1 - m2
    elif op == "*":
        result == m1*m2
    elif op == "/":
        result = m1/m2
    elif op == "dot":
        result = m1.dot(m2)
    else:
        result = None
    return result


# In[1]:


def create_matrix_positive_definite(m,n,start,end):
    E = None
    flag = False
    while flag == False:
        E = create_matrix_random(m,n,start,end)
        for i in range(E.shape[0]):
            for j in range(i):
                E[j][i]=E[i][j]
            test = np.linalg.eigvalsh(E)
            flag = np.all(test>0)
    return E


# In[4]:


def gradient_descent_2(alpha,x,y,numIterations):
    # x =[[1x0],[1x1],[1,2]...]
    m = x.shape[0] # number of samples
    theta = np.ones(2)
    for iter in range(0,numIterations):
        hypothesis = np.dot(x,theta)
        # hypothesis = theta0+theta1.x~x.dot(theta)~(1x).dot(theta0 theta1)
        loss = hypothesis -y
        J = np.sum(loss**2)/(2*m)#cost
        print('iter %s|J:%.3f'%(iter,J))
        theta0_prime = np.sum(loss)/m
        theta1_prime = np.sum(loss*x[:,1])/m
        gradient = np.array([theta0_prime,theta1_prime])
        theta = theta-alpha* gradient # update
    return theta


# In[ ]:




