import progressbar
from PIL import Image
import numpy as np
import helper_functions
from os.path import isfile
import os

# Initialize Parameters for custumization
# The Dimension Of The Image that is to be considered a tile
params = {'Tile Dimensions': {'x': 10, 'y': 10}}
params = {
    # The idmensions of the large mimage to be processed
    'Main Image Dimensions': {'x': 1280, 'y': 640},
    # Directory of the Main image to be proceessed
    'Main Image Directory': '1280*640.jpeg',
    # Ignore this
    'Tile Dimensions': params['Tile Dimensions'],
    # Print Debug Output
    'Debug Mode': True,
    # how many images in each batch
    'Batch Size': 5,
    # size of small batches in each normal batch:
    'Micro Batch Size': 2,
    # How many times to repatedly train on the images in the Repition folder
    'Repetition Image Repitions': 10,
    # Directory of the images that it will be repeatedly trained on.
    'Compressed Repition Training Folder Directory': 'dataset/Repeated_Images',
    # directory containing jpg
    'Compressed Train Image Directory': 'dataset/train',
    # directory containing jpg
    'Compressed Evaluation Image Directory': 'dataset/eval',
    # directory containing h5 files used in training
    'Decompressed Train Image Directory': 'processed/train',
    # directory containing h5 files used in evaluation
    'Decompressed Evaluation Image Directory': 'processed/eval',
    # directory containing h5 files used for repition
    'Decompressed Repition Training Folder Directory': 'processed/Repeated_Images',
    # The Number of neurons in each layer
    'Layer Dimensions': [params['Tile Dimensions']['x'] * params['Tile Dimensions']['y'], 7000, 3000, 900, 3000, 7000,
                         params['Tile Dimensions']['x'] * params['Tile Dimensions']['y']],
    # the neural networks optimizer
    'Network Optimizer': 'adam',
    # the neural networks loss function
    'Network Loss': 'mse',
    # what to measure when training the networks
    'Network Metrics': ['accuracy'],
    # The Activation Function Use After Each Layer
    'Network Activation Function': 'sigmoid',
    # Should The Program Load The Previoulsy Saved Model
    'Should Load Model': False,
    # Where to load the model from
    'Path To Model To Load': 'savedModel.h5',
    # Where to save the model
    'Path To Export Model To': 'savedModel.h5',
    # Where To Save Evaluations
    'Evaluation Save Directory': 'output',
    # name of the saved neural network diagram
    'Model Image Filename': 'model.png'
}


def cut_up_image(params):
    print ''
    print 'making tiles'
    # create in witch to store the tiles
    os.makedirs(params['Decompressed Train Image Directory'] + '/tiles')
    # the big image that is to be tiled
    mainImage = Image.open(params['Main Image Directory'])

    # checks if the image is the right diensions o that it can be tiled
    if (params['Main Image Dimensions']['x'] % params['Tile Dimensions']['x'] != 0) or (
                    params['Main Image Dimensions']['y'] % params['Tile Dimensions']['y'] != 0):
        # if it is not the right dimesnions it will notify the user
        print 'this image is not tileable'

    # the x width of a tile
    tileXsize = params['Main Image Dimensions']['x'] / params['Tile Dimensions']['x']

    # the y height of a tile
    tileYsize = params['Main Image Dimensions']['y'] / params['Tile Dimensions']['y']

    # createds the 2d array that will store tht tiles
    tiles = np.zeros((params['Main Image Dimensions']['x'] / tileXsize,
                      params['Main Image Dimensions']['y'] / tileYsize, tileXsize, tileYsize, 1))
    # loops through all positions in the array
    pbar = progressbar.ProgressBar()
    for indX, colum in enumerate(pbar(tiles)):
        for indY, img in enumerate(colum):
            # the tile image
            temporary = mainImage.crop(
                (indX * tileXsize, indY * tileYsize, (indX + 1) * tileXsize, (indY + 1) * tileYsize))

            # converts image into an array for storage in the array images
            tiles[indX][indY] = helper_functions.PILimageToNumpy(temporary)
    print 'done with tiles'
    print ''
    print 'making tile layers'
    # array that will store the tiles ocnverted to 128*128 images
    imgLayers = np.zeros((tileXsize, tileYsize, params['Tile Dimensions']['x'], params['Tile Dimensions']['y'], 1))

    pbar = progressbar.ProgressBar()
    # will loop through all tile and creat the layers and splits them into mixed processes
    for pixelToExtractX in pbar(range(tileXsize)):
        for pixelToExtractY in range(tileYsize):
            # creates the array that will store an image made from one pixel of each tile
            singleLayer = np.zeros((params['Tile Dimensions']['x'], params['Tile Dimensions']['y'], 1))
            # goes throught each tile
            for tilePosX in range(len(tiles)):
                for tilePosY in range(len(tiles[tilePosX])):
                    # and adds the specified pixel so "singleLayer"
                    singleLayer[tilePosX][tilePosY] = tiles[tilePosX][tilePosY][pixelToExtractX][pixelToExtractY]
            # then adds 'singleLayer' to the main array
            imgLayers[pixelToExtractX][pixelToExtractY] = singleLayer

    print imgLayers.shape

    print 'done with tile layers'
    # then returns the final result
    return np.flip(imgLayers, 0)


def reconstitute_images(params, imgLayers):
    print ''
    print 'reconsituting image'
    # the x width of a tile
    tileXsize = params['Main Image Dimensions']['x'] / params['Tile Dimensions']['x']

    # the y height of a tile
    tileYsize = params['Main Image Dimensions']['y'] / params['Tile Dimensions']['y']

    # the array that will store the reconstituted image
    mainImage = np.zeros((params['Main Image Dimensions']['x'], params['Main Image Dimensions']['y'], 1))
    pbar = progressbar.ProgressBar()
    for x in pbar(range(tileXsize)):
        for y in range(tileYsize):
            processedTile = imgLayers[x][y]
            for tileX, column in enumerate(processedTile):
                for tileY, pixel in enumerate(column):
                    mainImage[tileX * tileXsize + x][tileY * tileYsize + y] = pixel

    print(mainImage.shape)

    return mainImage


print ''
print 'clearing folder'
# the try catch is incase the folder "clearFolder" is told to empty is already empty
try:
    helper_functions.clearFolder(params['Decompressed Train Image Directory'])
except AttributeError:
    pass
print 'folder cleared'

helper_functions.saveImg(np.rot90(reconstitute_images(params, cut_up_image(params)), k=-1), "reconstitute.png",
                         {'x': 640, 'y': 1280})
