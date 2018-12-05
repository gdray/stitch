from pymodis import downmodis
import glob
from osgeo import gdal
import subprocess
import os
import rasterio
import numpy as np 
from pyproj import Proj, transform
import calendar
import glob
import skimage.io as skio
import matplotlib.pyplot as plt
import skimage.color as skc


def main():
    downloadFiles("2018-10-10")
    process()
    # print(res.shape)
    # plt.imsave('test.png', res)

    # join images together
    res = join()
    # get mask that predicts clouds (1 if cloud, 0 else)
    cloudMask = cloudDetector(res)
    # join mask and clouds to remove clouds from image
    resClouds = res & ~cloudMask

    plt.imsave('test.png', res)
    plt.imsave('clouds.png', cloudMask)
    plt.imsave('resWithClouds.png', resClouds)

    files = glob.glob("./data/stacked*.tif")
    transformFiles(files)

def downloadFiles(day):
    # Variables for data download
    dest = "data/" # This directory must already exist BTW
    tiles = ['h14v00', 'h15v00', 'h16v00', 'h17v00', 'h18v00', 'h19v00', 'h20v00', 'h21v00', 'h11v01', 'h12v01', 'h13v01', 'h14v01', 'h15v01', 'h16v01', 'h17v01', 'h18v01', 'h19v01', 'h20v01', 'h21v01', 'h22v01', 'h23v01', 'h24v01']

    # enddate = "2018-10-11" # The download works backward, so that enddate is anterior to day=
    product = "MOD09GA.006"

    # Instantiate download class, connect and download
    modis_down = downmodis.downModis(destinationFolder=dest, tiles=tiles, user="lukealvoeiro", password="Burnout1", today=day, delta=1, product=product)
    modis_down.connect()
    modis_down.downloadsAllDay()

    # Check that the data has been downloaded
    MODIS_files = glob.glob(dest + '*.hdf')
    print(MODIS_files)

def process(convertToTiff=True):
    owd = os.getcwd()
    dest = "data/" 
    MODIS_files = [i[5:] for i in glob.glob(dest + '*.hdf')]
    os.chdir(dest)

    bands = [11,13,14, 17]  # bands to download
    
    if(convertToTiff): getTiffFiles(MODIS_files, bands)
    combineChannels()   # combine channels into single image         

def getTiffFiles(MODIS_files, bands):
    with open("list_modis_files.txt", "w") as output_file:
        for file in MODIS_files:
            for band in bands:
                sds = gdal.Open(file, gdal.GA_ReadOnly).GetSubDatasets()
                src = gdal.Open(sds[band][0])

                filename =  "band" + str(band) + "_" + file.split(".")[2] +".tif"
                gdal.Translate(filename, src)
                output_file.write(filename + "\n")
    
def combineChannels():
    """
    Given a list of n files, stacks them on top of each other producing a geotiff with
    the same metadata, but with all bands available.
    """
    
    with open("list_modis_files.txt", "r") as f:
        file_list = f.read().splitlines()
    
    stacked_files = list(set([i[7:13] for i in file_list]))
    for file_start in stacked_files:
        files = glob.glob("*" + file_start + '.tif')
        dont_include = glob.glob("stacked*.tif")
        files = sorted(list(set(files).difference(dont_include)))
        
        print("Combining...", files)
        with rasterio.open(files[0]) as src0:
            meta = src0.meta

        meta.update(count = len(files))

        # Read each layer and write it to stack
        with rasterio.open("stacked_" + file_start + ".tif", 'w', **meta) as dst:
            
            for id, layer in enumerate(files, start=1):
                with rasterio.open(layer) as src1:
                    dst.write_band(id, src1.read(1))

def getCoordinatesFromIndices(filename, x_index, y_index):
    """
    Given a file and indices x and y, get their position in real 
    latitude and longitude coordinates
    """
    if(len(x_index) != len(y_index) or len(y_index) == 0 or len(x_index) == 0):
        return []

    # Read raster
    with rasterio.open(filename, 'r') as src:
        if(src.crs.is_valid):
            # Determine Affine object and CRS projection
            trans = src.transform
            inProj = Proj(src.crs)
            outProj = Proj(init='epsg:4326')

            res = []
            for i in range(len(x_index)):
                curr_x, curr_y = x_index[i], y_index[i]
                # Determines East/Northing 
                x, y = rasterio.transform.xy(trans, curr_x, curr_y)
                # Convert these to latitude / longitude
                tmp = transform(inProj,outProj,x,y)
                res.append(tmp)
            
            return res   

def getIndicesOfCoordinate(filename, x_coord, y_coord):
    """
    Given an image file and latitude and longitude coordinates x and y, 
    get their corresponding indices in the image
    """
    
    if(len(x_coord) != len(y_coord) or len(y_coord) == 0 or len(x_coord) == 0):
        return []
    
    # Read raster
    with rasterio.open(filename, 'r') as src:
        # Determine Affine object and CRS projection
        trans = src.transform
        inProj = Proj(init='epsg:4326')
        outProj = Proj(src.crs)

        res = []
        for i in range(len(x_coord)):
            curr_x, curr_y = x_coord[i], y_coord[i]
            # Determines East/Northing 
            x, y = transform(inProj, outProj, curr_x, curr_y)
            # Convert the point to indices in the image
            tmp = rasterio.transform.rowcol(trans, x, y)
            res.append(tmp)
        
        return res

def produceStackedArray(filename):
    """
    Given a image file, combines all the bands into a numPy array
    where each pixel corresponds to a list of n values, where n is the number
    channels in the image provided
    """
    with rasterio.open(filename, 'r') as src:
        count = src.count
        base = src.read(1)
        for band_num in range(2, count+1):
            base = np.dstack((base, src.read(band_num)))
        return base

def getHorizontalBorderCoords(filename_imgA, filename_imgB):
    imgA = skio.imread(filename_imgA)
    imgB = skio.imread(filename_imgB)
    if(imgA.shape != imgB.shape): return False
    rowIndicesToCheck = list(range(imgA.shape[0]))
    imgACoords = np.array(getCoordinatesFromIndices(filename_imgA, rowIndicesToCheck, [imgA.shape[1]]*len(rowIndicesToCheck)))
    imgBCoords = np.array(getCoordinatesFromIndices(filename_imgB, rowIndicesToCheck, [0]*len(rowIndicesToCheck)))
    return imgACoords, imgBCoords

def getVerticalBorderCoords(filename_imgA, filename_imgB):
    imgA = skio.imread(filename_imgA)
    imgB = skio.imread(filename_imgB)
    if(imgA.shape != imgB.shape): return False
    colIndicesToCheck = list(range(imgA.shape[1]))
    imgACoords = np.array(getCoordinatesFromIndices(filename_imgA, [imgA.shape[0]]*len(colIndicesToCheck), colIndicesToCheck))
    imgBCoords = np.array(getCoordinatesFromIndices(filename_imgB, [0]*len(colIndicesToCheck), colIndicesToCheck))
    return imgACoords, imgBCoords

def checkDifference(filename_imgA, filename_imgB, typeDiff='H'):
    """
    Checks if there is a difference between the two files starting and ending points. Make sure images are placed in the correct order
    typeDiff can be either 'H' or 'V' (Horizontal or Vertical)
    """
    if(typeDiff == 'H'): coordsA, coordsB = getHorizontalBorderCoords(filename_imgA, filename_imgB)
    else: coordsA, coordsB = getVerticalBorderCoords(filename_imgA, filename_imgB)
    diff = coordsA - coordsB
    print(np.average(diff, axis=0))

def join():
    rows = list(range(0, 2))
    res = None
    itr = True
    for row in rows:
        files = sorted(glob.glob("./data/stacked*" + str(row) + ".tif"))
        base = processImageBeforeStitching(skio.imread(files[0]))
        for col_img in files[1:]:
            tmp = processImageBeforeStitching(skio.imread(col_img))
            base = np.hstack((base, tmp))
        if(itr): 
            filler = np.resize(np.array([0, 0, 0]), (2400, 7200, 3))
            base = np.hstack((base, filler))
            res = np.hstack((filler, base))
            itr = False
        else: res = np.vstack((res, base))
    
    print(res.shape)
    return res

def outputPredictedImage(inputFilename, outputFilename, outputImage, rows = 4800, cols = 4800):
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outputFilename, rows, cols, 1, gdal.GDT_Byte)
    base = gdal.Open(inputFilename)
    outdata.SetGeoTransform(base.GetGeoTransform())
    outdata.SetProjection(base.GetProjection())
    outdata.GetRasterBand(1).WriteArray(outputImage)
    outdata = None
    base = None

def processImageBeforeStitching(img):
    img[img == -28672] = -100
    img += 100
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = img.astype(np.uint8)
    tmp = img[:,:,1]
    img[:,:,1] = img[:,:,2]
    img[:,:,2] = tmp
    return img

def cloudDetector(img):
    rows, cols, bands = img.shape
    clouds = np.zeros((rows, cols, bands))
    #clouds[(img > 178) & (img < 204)] = 255
    #res = skc.rgb2gray(clouds) > 0

    clouds[(img[:,:,4] > 255*0.35)] = 255
    cloudMask = (skc.rgb2gray(clouds) > 0)
    return cloudMask

def transformFiles(files):
    outfile = 'mosaic.vrt'
    outfile2 = 'testProject.tif'
    gdal.BuildVRT(outfile, files)
    opt= gdal.WarpOptions(format = 'GTiff', srcSRS= '+proj=sinu',
    dstSRS = 'EPSG:3413')
    gdal.Warp(outfile2, outfile)

if __name__ == '__main__':
    main()