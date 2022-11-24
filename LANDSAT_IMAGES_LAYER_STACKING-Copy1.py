#NDVI
import rasterio
from matplotlib import pyplot
from osgeo import gdal
import numpy as np
#reading the file with specified path
LST = rasterio.open('D:/rambabu_r@nrsc.gov.in/STUDY/MTECH/IIST GeoInformatics Text books/1 SEM/SCGA&LAB/LABS/SCGA_LAB_4/stack_LST.tiff')
#reading meta data
LST.meta
#visualising the band 4 in satellite image
pyplot.imshow(LST.read(4), cmap='terrain')

# Open the fourth band in our image - NIR here
nir = LST.read(4).astype(float)
red = LST.read(3).astype(float)

dinom = nir+red
numer = nir-red
np.seterr(divide='ignore', invalid='ignore')
ndvi = np.where(dinom==0.0, 0.0, numer/dinom)

ndvi.flatten().max() # for checking the NDVI values (-1 to +1)
ndvi.flatten().min()

pyplot.imshow((ndvi), cmap='terrain')





