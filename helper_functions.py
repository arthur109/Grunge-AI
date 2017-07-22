from os.path import isfile, join
from os import listdir
import scipy.misc
from keras.layers import Dense, Input, Reshape
from keras.models import Model
import h5py
import numpy as np
from PIL import Image
from keras.preprocessing import image
import progressbar
import shutil
import os


# input is a directory from which to get the files so it can return a list containg the paths of all the h5's inside it
def loadH5sFromFolder(directory):
    return sorted([join(directory, f)
                   for f in listdir(directory)
                   if (isfile(join(directory, f)) and ("part" in f))])  # returns paths of all h5 files


# input is a directory from which to get the files so it can return a list contaning the paths of all the images inside it
def loadIMGsFromFolder(directory):
    return sorted([join(directory, f)
                   for f in listdir(directory)
                   if (isfile(join(directory, f)) and ("img" in f))])  # returns paths of all images in folder


# creates the neural network
def makeModel(params):
    # Create An Input Layer From The Image Dimensions
    inputLayer = Input(shape=(params['Tile Dimensions']['x'], params['Tile Dimensions']['y'], 1))
    # Flatten Out The Image
    prevLayer = Reshape((params['Tile Dimensions']['x'] * params['Tile Dimensions']['y'],))(inputLayer)

    # Create All The Layers
    for layerDim in params['Layer Dimensions'][:-1]:
        prevLayer = Dense(layerDim, activation=params['Network Activation Function'])(prevLayer)
    prevLayer = Dense(params['Layer Dimensions'][-1], activation='tanh')(prevLayer)
    # Reshapes The Output Back Into An Image (from 1d to 2d)
    outputLayer = Reshape((params['Tile Dimensions']['x'], params['Tile Dimensions']['y'], 1))(prevLayer)

    # Create The Network
    model = Model(inputs=inputLayer, outputs=outputLayer)
    # Setup The Network
    model.compile(optimizer=params['Network Optimizer'],
                  loss=params['Network Loss'],
                  metrics=params['Network Metrics'])
    # if the program shoud load the model
    if params['Should Load Model']:
        print 'loading model...'
        # then it loads the model
        model.load_weights(params['Path To Export Model To '])
        print 'MODEL LOADED'

    return model


# takes in the image array, the filename of the image to be saved, the dimensions of the image as {'x':x,'y':y}, and params
def saveImg(imgInput, filename, dimensions):
    # Make sure the array is the right shape
    imgInput = imgInput.reshape((dimensions['x'], dimensions['y'], 1))
    # Make the image have 3 channels
    img = changeChannels1to3(imgInput, dimensions)
    # Save the image to the specified filename

    scipy.misc.toimage(img, cmin=0.0, cmax=1).save(filename)


# performs transformations on image before saving it because sometimes problems are caused like distortion or incorrect rotation if they are not applied
def saveImg2(imgInput, filename, dimensions):
    imgInput = imgInput.reshape((dimensions['x'], dimensions['y'], 1))
    saveImg(np.rot90(imgInput, k=-1), filename, {'x': dimensions['y'], 'y': dimensions['x']})


# converts an image with only one channel(black and white) and converts it to full rgb(uduplicates the channel)
# so it can be properly exported to an image
def changeChannels1to3(imgInput, dimensions):
    # Create A Blank Image With 3 Channels
    img = np.zeros((dimensions['x'], dimensions['y'], 3), dtype=np.float32)
    # Loop Through Each Row In the image
    for h, row in enumerate(imgInput):
        # Loop Through each column in each row
        for i, col in enumerate(row):
            # Set All 3 Channels to the original BW value
            for channel in col:
                img[h][i][0] = channel
                img[h][i][1] = channel
                img[h][i][2] = channel
    return img


# a function that takes the path of an h5 file and return the conetents of the file in an array
def loadBatch(path):
    # opens the file in reading mode, hence the 'r'
    batchH5F = h5py.File(path, 'r')
    # gets all the data inside it
    batch = batchH5F['data'][:]
    # closes the file
    batchH5F.close()
    # returns the array of its contents
    return batch


# a function that takes in a 2d array in format:
# [
#    [input folder directory, output folder directory],
#    [input folder directory, output folder directory],
#    ...
#    for all other things that must convert from images to h5.py files
# ]
# (also takes in params so it know the batch size, and evaluation batch size)
def make_batches(params, inputOutputLocations):
    # goes through every arra in the inputOutputLocations array
    for pathsToImages in inputOutputLocations:
        # gets the directorys of the input and output folders
        inputFolder = pathsToImages[0]
        outputFolder = pathsToImages[1]
        # clears all previous data in the output folder using the function created below
        clearFolder(outputFolder)
        # uses function loadIMGsFromFolder(that was created above) to get a list of all the paths of the images in the input folder
        imagePaths = loadIMGsFromFolder(inputFolder)
        # creats an empty list that will contain all the images
        imagesAsList = []
        # adds a progress bar and titles it (so we know what its doing)
        print 'loading images...'
        pbar = progressbar.ProgressBar()
        # loops therough all the image paths and displays progress bar (that is done with the pbar())
        for i in pbar(range(len(imagePaths))):
            # gets the image inan array using the function 'image_to_array' created below and stores it in the variable 'img'
            img = img_to_array(imagePaths[i], params['Tile Dimensions'])
            # checks if there were no errors with the conversion process
            if not (type(img) == 'NoneType'):
                try:
                    # trys to append image
                    imagesAsList.append(img)
                except ValueError:
                    # if there is an error in apending it it prints 'the images path + is bad'
                    print imagePaths[i] + " is bad"
            else:
                # also prints 'the images path + is bad'
                print imagePaths[i] + " is bad"
                # creats new np array which will contain all images (and will replace imagesAsList)
        images = np.zeros((len(imagesAsList), params['Tile Dimensions']['x'], params['Tile Dimensions']['y'], 1))
        # adds the images
        for i in range(len(images)):
            images[i] = imagesAsList[i]

        # ex: you have 30 images, batch size = 3
        # number of images/batch size = 30/3 = 10
        # range(1,10) = np.array[1,2,3,4,5,6,7,8,9] (doesn't include last number)
        # that np array * batch size = [1*3,2*3,3*3,...] = [3,6,9,12,15,18,21,24,27,30]
        # and then splits it on that array (splits function on numbers in the array as index positions returning an array of arrays)
        # all in that line of code, below
        batches = np.split(images, (
            np.array(range(1, int(len(images) / params['Batch Size']))) * params['Batch Size']).tolist())

        # goes through each array in array and writes it to a h5 file
        for i, batch in enumerate(batches):
            # gives it a proper name and path(output folder), 'w' means writing not reading the file
            h5f = h5py.File(outputFolder + "/part-" + str(i).zfill(7) + ".h5", "w")
            # adds data to file
            h5f.create_dataset('data', data=batch)
            # closes the file. Remeber to alwayse put this at the end
            h5f.close()
        images = None


# a function that takes a directory and deletes all things inside it.
def clearFolder(folder):
    # laods an array with paths of all files
    allFiles = [folder + "/" + f for f in listdir(folder)]
    # goes throught each path and delets the coresponding file
    pbar = progressbar.ProgressBar()
    for f in pbar(allFiles):
        try:
            os.remove(f)
        except OSError:
            shutil.rmtree(f)


# a function that takes in a path and the image dimesions and outputs an array containg all the black and white pixel values in range 0 to 1
def img_to_array(imgPth, dimensions):
    img = None
    try:
        # tries to load image
        img = image.load_img(imgPth, target_size=(dimensions['x'], dimensions['y']))
        # converts image to array
        img = image.img_to_array(img)
        # scales rgb values form 0-1
        img *= 1. / 255
        # makes it float 16 so it takes up less space
        img = img.astype(np.float16)
        # delets g and b rgb values to make balck and white
        img = changeChannels3to1(img)
    except IOError:
        # if it failed to load the image it prints image path is bad
        print imgPth + " is bad"
        # and returns nothing
        return None
    # if not, returns the image array
    return img


# Strips a numpy array representing and image of 2 of its channels make sure the image is already in greyscale
def changeChannels3to1(img):
    # Removes channel 2 on the 3rd axis, -1 on both because it starts at 0
    img = np.delete(img, 1, 2)
    img = np.delete(img, 1, 2)
    return img


# Converts A PIL Image to a numpy array and rotates it 90 degrees because other wise it is sideways for and unknow reason
def PILimageToNumpy(img):
    return changeChannels3to1(np.rot90(np.asarray(img))) * (1. / 255)


# Separates and image into tiles and returns them as an array of shape (num tiles x, num tiles y, size of tile x, size of tile y, 1 channel)
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
            tiles[indX][indY] = PILimageToNumpy(temporary)
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
            imgLayers[pixelToExtractX][pixelToExtractY] = np.flip(singleLayer, 0)

    print imgLayers.shape

    print 'done with tile layers'
    # then returns the final result
    return imgLayers

# reconverts the processed/commpressed image format into a normal image that you can see
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
    # goes through each 128*128 tile
    for x in pbar(range(tileXsize)):
        for y in range(tileYsize):
            #stores the 128*128 in the array processedTile
            processedTile = imgLayers[x][y]
            # loops through each pixel in processedTile
            for tileX, column in enumerate(processedTile):
                for tileY, pixel in enumerate(column):
                    # and assignes it to its Corresponding position which is (which pixel its on * the 5*5 tile size + which layer its doing (top right corner, ect...))
                    mainImage[tileX * tileXsize + x][tileY * tileYsize + y] = pixel

    return mainImage
