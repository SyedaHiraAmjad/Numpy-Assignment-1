#!/usr/bin/env python
# coding: utf-8

# # Numpy Functions 

# ## Importing (Keys and Import)

# In[1]:


pip install numpy


# In[2]:


pip install --upgrade pip


# In[3]:


import numpy as np


# In[5]:


pip install --upgrade ipython jupyter


# ## Creating Arrays

# In[7]:


np.array([1,2,3])


# In[8]:


np.array([(1,2,3),(4,5,6)])


# In[9]:


#1-D array#
np.zeros(3)


# In[10]:


np.ones((3,4))


# In[11]:


np.eye(5)


# In[12]:


np.linspace(0,100,6)


# In[13]:


np.arange(0,10,3)


# In[14]:


np.full((2,3),8)


# In[15]:


np.random.rand(4,5)


# In[16]:


np.random.rand(6,7)*100


# In[17]:


np.random.randint(5,size=(2,3))


# ## Inspecting Properties

# In[23]:


arr1 = [1,2,3,4,5,6,7,8]
nums = np.array ([1,2,3,4,5,6,7,8])
type(nums)


# In[24]:


arr1 = np.arange(10)
arr1


# In[25]:


#Array Size#
arr1.size


# In[26]:


#Array Shape#
arr1.shape


# In[27]:


#Array Type#
arr1.dtype


# In[29]:


#Array Type#
arr.astype


# In[30]:


#Array tolist#
arr1.tolist()


# In[31]:


np.info(np.eye)


# In[32]:


#d-type parameters#
arr2 = np.array([1,2,3,4,5,6,7,8,9,10], dtype = complex)
arr2


# In[33]:


#n-dimension#
arr1 = np.array([(1,2,3,4),(5,6,7,8)])
arr1.ndim


# ## Dimensions

# In[34]:


#1-Dimesion#
arr1 = np.arange(5)
arr1


# In[42]:


#2-Dimension#
arr3 = np.arange(10).reshape
arr3


# In[43]:


#3-Dimension#
arr4 = np.arange(24).reshape(2,3,4)
arr4


# In[49]:


#Multi-dimensions#
arr5 = np.array([(1,2,3),(4,5,6)])
arr5


# In[50]:


#n-dimension#
arr1 = np.array([(1,2,3,4),(5,6,7,8)])
arr1.ndim


# In[52]:


#Minimum-Dimensions#
arr6 = np.array([0,1,2,3,4,5,6,7,8,9,10],ndmin = 2)
arr6


# In[85]:


#2-Dimension zero matrix#
arr = np.zeros((2,3))
arr


# ## Copying / Sorting / Reshaping

# In[54]:


#Copy array#
np.copy(arr1)


# In[57]:


#View array#
arr1.view('int64')


# In[60]:


#Sort array#
arr1.sort()


# In[61]:


arr1.sort(axis=0)


# In[64]:


#Transpose of Array#
arr1.T


# ## Adding / Removing /Indexing elements

# In[76]:


#Inserting values#
np.insert(arr1,8,[9,10])


# In[83]:


#Indexing values#
arr7 = np.array([1,2,3,4,5,6,7,8])
arr7[[0,1]]


# In[84]:


#Identity matrix#
np.identity(4)


# ## Combining / Splitting

# In[87]:


#Concatenating arrays (axis=0)#
arr1 = [0,1,2,3,4]
arr2 = [5,6,7,8,9]
arr = np.concatenate((arr1,arr2),axis=0)
arr


# In[90]:


#Concatenating arrays (axis=1)#
arr3 = ([(1,2,3),(4,5,6)])
arr4 = ([(5,6,7),(8,9,10)])
arr = np.concatenate((arr3,arr4),axis=1)
arr


# In[97]:


#Splitting array#
np.array_split(arr1,3)


# In[100]:


#Horizontal splitting array#
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])

new_arr = np.hsplit(arr, 3)
new_arr


# ## Indexing / Slicing / Subsetting

# In[101]:


arr[5]


# In[104]:


arr[1] = 4


# In[107]:


#Indexing#
arr = np.array([1,2,3,4,5,6,7,8])
arr[[0,1]]


# In[108]:


#log-space array#
arr = np.logspace(3,8,num=10,base=1000.0,dtype=float)
arr


# In[109]:


#Diagonal Array#
np.diag(np.arange(0,8,1))


# In[113]:


np.diag(np.diag(np.arange(12).reshape((4,3))))


# In[114]:


#Vertical stack#
arr1 = np.array([(1,2,3,4),(5,6,7,8)])
arr2 = np.array([(9,10,11,12),(13,14,15,16)])
np.vstack((arr1,arr2))


# In[115]:


#Horizontal stack#
np.hstack((arr1,arr2))


# In[116]:


#Linear spacing#
arr = np.linspace(1,3,10)
print(arr)


# In[118]:


#Slicing#
arr = np.array([(1,2,3,4),(5,6,7,8)])
print(arr[0:,1])


# In[146]:


#Slicing in multi-dimension#
arr = np.array([(2,3),(4,5),(7,8)])
arr[0:2,1]


# In[123]:


(arr1<3)&(arr2>5)


# In[125]:


~arr2


# In[127]:


arr2[arr2<5]


# In[128]:


#Boolean Method#
bool_arr = np.array([1,.5,0,'aa',''],dtype=bool)
print(bool_arr)


# In[129]:


#Filtering values (odd number)#
arr = np.array([0,1,2,3,4,5,6,7,8,9,10])
arr[arr%2==1]


# In[130]:


#Generating Random Number#
np.random.rand(3,3)


# In[131]:


#Random number in a given range#
np.random.randint(2,size=5)


# In[132]:


np.random.randint(15,size=(2,3))


# In[133]:


#last value#
arr1[-1]


# ## Scalar Maths

# In[134]:


#Adding value#
np.add(arr1,1)


# In[136]:


#Adding in matrix#
arr2 = np.array([(1,2,3),(4,5,6)])
arr3 = np.array([(7,8,9),(10,11,12)])
arr2+arr3


# In[137]:


#Subtracting value#
np.subtract(arr1,2)


# In[138]:


#Subtracting in matrix#
arr2-arr3


# In[139]:


#Multiplying value#
np.multiply(arr1,3)


# In[140]:


#Multiplying in matrix#
arr2*arr3


# In[141]:


#Dividing value#
np.divide(arr1,4)


# In[142]:


#Dividing in matrix#
arr2/arr3


# In[143]:


#Squareroot#
np.sqrt(arr1)


# In[144]:


#Minimum values#
arr1.min()


# In[145]:


#Maximum values#
arr1.max()


# In[147]:


#Power#
np.power(arr2,5)


# ## Vector Maths

# In[148]:


#Adding#
np.add(arr2,arr3)


# In[150]:


#Subtracting#
np.subtract(arr2,arr3)


# In[151]:


#Multiplying#
np.multiply(arr2,arr3)


# In[152]:


#Dividing#
np.divide(arr2,arr3)


# In[153]:


#Power#
np.power(arr2,arr3)


# In[154]:


#Checking (equal or not)#
np.array_equal(arr2,arr3)


# In[156]:


#Square-root#
np.sqrt(arr2)


# In[157]:


# *Trignometric Functions* #
#Sin#
np.sin(arr1)


# In[158]:


#Cos#
np.cos(arr2)


# In[159]:


#Tan#
np.tan(arr3)


# In[160]:


#Logarithm#
np.log(arr1)


# In[161]:


#Exponential#
np.exp(arr2)


# In[162]:


#Absolute value#
np.abs(arr3)


# In[163]:


#Ceiling value#
np.ceil(arr1)


# In[164]:


#Floor value#
np.floor(arr2)


# In[165]:


#Round-off value#
np.round(arr3)


# In[166]:


#Reciprocal#
np.reciprocal(arr1)


# In[167]:


#Remainder#
np.remainder((arr2),3)


# In[168]:


#Sign#
np.sign(arr1)


# ## Statistics

# In[169]:


#Mean value#
np.mean(arr2,axis=0)


# In[170]:


arr2.sum()


# In[171]:


#Minimum value#
arr.min()


# In[172]:


#Maximum value (axis=0)#
arr.max(axis=0)


# In[173]:


#Variance#
np.var(arr2)


# In[174]:


#Standard deviation#
np.std(arr2,axis=1)


# In[178]:


#Coefficient of Corelation#
np.corrcoef(arr2)


# In[ ]:




