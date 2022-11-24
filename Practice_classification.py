#!/usr/bin/env python
# coding: utf-8

# In[1]:


#RAMBABU R 
#SC22M075
#IIST/Geoinformatics
#Remote sensing lab

from osgeo import gdal
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from spectral import*


# In[2]:


#the data set contains 425 bands of AVIRIS NG of ICRISAT HYDERABAD AREA
#performing the unsupervised k means clustering iterations using spectral library
data = rasterio.open('ICRISAT_SUBSET_AVIRIS_recta.tiff')


# In[3]:


crs = data.crs
col = data.width
rows = data.height
bands = data.count
print(crs,col,rows,bands)


# In[4]:


data_matrix = data.read()


# In[18]:


data_matrix


# In[19]:


data_matrix.T


# In[20]:


data_matrix.T.shape
#columns, Rows, no.of bands


# In[21]:


# for unsupervised classification the k-means algorithm on the image and create 20 clusters, using a maximum of 50 iterations, 
#here we are running the maximum 30 iterations only.
(m, c) = kmeans(data_matrix.T, 20, 30)


# In[22]:


# for displaying the image 
imshow(classes=m)


# In[32]:


for i in range(c.shape[0]):
    plt.plot(c[i])
    plt.grid()


# In[11]:


#for calculating NDVI
#defining the bands
nir = data_matrix[75,:,:]
red = data_matrix[55,:,:]


# In[12]:


dinom = nir+red
numer = nir-red
np.seterr(divide='ignore', invalid='ignore') 
savi = np.where(dinom==0.0, 0.0, numer/dinom)


# In[13]:


#SAVI
dinom = nir+red+0.5
numer = (nir-red)*1.5
np.seterr(divide='ignore', invalid='ignore') 
ndvi = np.where(dinom==0.0, 0.0, numer/dinom)


# In[14]:


plt.figure()
plt.imshow(savi)


# In[15]:


defining the the out put mata data format for writing the NDVI
# here we are defining the meta data format as same as input data file
out_meta = data.meta.copy()

out_meta.update({'driver':'GTiff',
                 'width':data.shape[1],
                 'height':data.shape[0],
                 'count':1,
                 'dtype':'float64',
                 'crs':data.crs, 
                 'transform':data.transform,
                 'nodata':0})


# In[16]:


#now how to save the ndvi data apply the above parameters
with rasterio.open(fp=r'ICRISAT_NDVI.tif', # outputpath_name
             mode='w',**out_meta) as dst:
             dst.write_band(1,ndvi) # the numer one is the number of bands


# In[ ]:





# In[30]:


plt.figure()
plt.imshow(ndvi)


# In[8]:


#defining the the out put mata data format for writing the NDVI
# here we are defining the meta data format as same as input data file
out_meta = data.meta.copy()

out_meta.update({'driver':'GTiff',
                 'width':data.shape[1],
                 'height':data.shape[0],
                 'count':1,
                 'dtype':'float64',
                 'crs':data.crs, 
                 'transform':data.transform,
                 'nodata':0})


# In[38]:


#now how to save the ndvi data apply the above parameters
with rasterio.open(fp=r'ICRISAT_NDVI.tif', # outputpath_name
             mode='w',**out_meta) as dst:
             dst.write_band(1,ndvi) # the numer one is the number of bands


# In[40]:


#supervised classification 
trained = rasterio.open('avi.tiff')
data = rasterio.open('ICRISAT_SUBSET_AVIRIS_recta.tiff')


# In[44]:


t = trained.read()
dataset = data.read()


# In[49]:


classes = create_training_classes(dataset.T, t.T)


# In[50]:


#Gaussian Maximum Likelihood Classification
gmlc = GaussianClassifier(classes)


# In[52]:


clmap = gmlc.classify_image(dataset.T)


# In[53]:


imshow(classes=clmap)


# In[54]:


# supervised classification using spectral angles
classes = create_training_classes(dataset.T, t.T, True)


# In[55]:


means = np.zeros((len(classes), dataset.T.shape[2]), float)


# In[56]:


for (i, c) in enumerate(classes):
       means[i] = c.stats.mean


# In[57]:


angles = spectral_angles(dataset.T, means)


# In[58]:


clmap = np.argmin(angles, 2)


# In[62]:


imshow(classes=clmap)


# In[86]:


nir = data_matrix[75,:,:]
red = data_matrix[55,:,:]
green = data_matrix[35,:,:]


# In[89]:


falsecolor = rasterio.open('C:/Users/RAMBABU/Geopandas/falsecolor_AVIRIS.tiff','w',driver='Gtiff',
                    width= data.width, height= data.height,
                    count=3,
                    crs = data.crs,
                    transform = data.transform,
                    dtype = 'uint16'
                    )
falsecolor.write(green(1),3)#blue
falsecolor.write(red(1),2)#green
falsecolor.write(nir(1),1)#red
falsecolor.close()


# In[76]:


#creating true colour composite image 
Truecolor = rasterio.open('C:/Users/RAMBABU/Geopandas/Truecolor_AVIRIS.tiff','w',driver='Gtiff',
                    width= data.width, height= data.height,
                    count=3,
                    crs = data.crs,
                    transform = data.transform,
                    dtype = 'uint16'
                    )
Truecolor.write(blue,3)#blue
Truecolor.write(green,2)#green
Truecolor.write(red,1)#red
Truecolor.close()


# In[ ]:




