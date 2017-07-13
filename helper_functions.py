from os.path import isfile, join
from os import listdir
import scipy.misc
from keras.layers import Dense, Activation, Flatten, Input, Reshape
from keras.models import Model
import h5py
import numpy as np
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


def makeModel(params):
    # Create An Input Layer From The Image Dimensions
    inputLayer = Input(shape=(params['Image Dimensions']['x'], params['Image Dimensions']['y'], 1))
    # Flatten Out The Image
    prevLayer = Reshape((params['Image Dimensions']['x'] * params['Image Dimensions']['y'], ))(inputLayer)

    # Create All The Layers
    for layerDim in params['Layer Dimensions'][:-1]:
        prevLayer = Dense(layerDim, activation=params['Network Activation Function'])(prevLayer)
    prevLayer = Dense(params['Layer Dimensions'][-1], activation='tanh')(prevLayer)
    # Reshapes The Output Back Into An Image (from 1d to 2d)
    outputLayer = Reshape((params['Image Dimensions']['x'], params['Image Dimensions']['y'], 1))(prevLayer)

    # Create The Network
    model = Model(inputs=inputLayer, outputs=outputLayer)
    # Setup The Network
    model.compile(optimizer=params['Network Optimizer'],
                  loss=params['Network Loss'],
                  metrics=params['Network Metrics'])

    return model

def makeModelV2(params):
    print "hi"

def saveImg(imgInput, filename, params):
    # Make sure the array is the right shape
    imgInput = imgInput.reshape((params['Image Dimensions']['x'], params['Image Dimensions']['y'], 1))
    # Make the image have 3 channels
    img = changeChannels1to3(imgInput, params)
    # Save the image to the specified filename
    scipy.misc.toimage(img, cmin=0.0, cmax=1).save(filename)

# converts an image with only one channel(black and white) and converts it to full rgb(uduplicates the channel)
# so it can be properly exported to an image


def changeChannels1to3(imgInput, params):
    # Create A Blank Image With 3 Channels
    img = np.zeros((params['Image Dimensions']['x'], params['Image Dimensions']['y'], 3), dtype=np.float32)
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
            img = img_to_array(imagePaths[i], params['Image Dimensions'])
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
        images = np.zeros((len(imagesAsList), params['Image Dimensions']['x'], params['Image Dimensions']['y'], 1))
        # adds the images
        for i in range(len(images)):
            images[i] = imagesAsList[i]

        # ex: you have 30 images, batch size = 3
        # number of images/batch size = 30/3 = 10
        # range(1,10) = np.array[1,2,3,4,5,6,7,8,9] (doesn't include last number)
        # that np array * batch size = [1*3,2*3,3*3,...] = [3,6,9,12,15,18,21,24,27,30]
        # and then splits it on that array (splits function on numbers in the array as index positions returning an array of arrays)
        # all in that line of code, below
        batches = np.split(images, (np.array(range(1, int(len(images) / params['Batch Size']))) * params['Batch Size']).tolist())

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
    for f in allFiles:
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
        img = np.delete(img, 1, 2)
        img = np.delete(img, 1, 2)
    except IOError:
        # if it failed to load the image it prints image path is bad
        print imgPth + " is bad"
        # and returns nothing
        return None
    # if not, returns the image array
    return img
