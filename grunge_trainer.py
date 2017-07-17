# Import Libraries
import os
from os.path import isfile

import numpy as np
import progressbar
from keras.utils import plot_model

import helper_functions

# Initialize Parameters for customization
# The Dimension Of The Image
params = {'Image Dimensions': {'x': 128, 'y': 128}}
params = {
    # Ignore this
    'Image Dimensions': params['Image Dimensions'],
    # Print Debug Output
    'Debug Mode': True,
    # how many images in each batch
    'Batch Size': 5,
    # size of small batches in each normal batch:
    'Micro Batch Size': 2,
    # How many times to repeatedly train on the images in the Repetition folder
    'Repetition Image Repetitions': 10,
    # Directory of the images that it will be repeatedly trained on.
    'Compressed Repetition Training Folder Directory': 'dataset/Repeated_Images',
    # directory containing jpg
    'Compressed Train Image Directory': 'dataset/train',
    # directory containing jpg
    'Compressed Evaluation Image Directory': 'dataset/eval',
    # directory containing h5 files used in training
    'Decompressed Train Image Directory': 'processed/train',
    # directory containing h5 files used in evaluation
    'Decompressed Evaluation Image Directory': 'processed/eval',
    # directory containing h5 files used for repetition
    'Decompressed Repetition Training Folder Directory': 'processed/Repeated_Images',
    # The Number of neurons in each layer
    'Layer Dimensions': [params['Image Dimensions']['x'] * params['Image Dimensions']['y'], 7000, 3000, 900, 3000, 7000,
                         params['Image Dimensions']['x'] * params['Image Dimensions']['y']],
    # the neural networks optimizer
    'Network Optimizer': 'adam',
    # the neural networks loss function
    'Network Loss': 'mse',
    # what to measure when training the networks
    'Network Metrics': ['accuracy'],
    # The Activation Function Use After Each Layer
    'Network Activation Function': 'sigmoid',
    # Should The Program Load The Previously Saved Model
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

# uses the function make_batches in the helper_functions folder to convert the images to h5 files in the batch sizes
# (also does it for the evaluation images)
helper_functions.make_batches(params, [
    # training
    [params['Compressed Train Image Directory'],
     params['Decompressed Train Image Directory']],
    # evaluation
    [params['Compressed Evaluation Image Directory'],
     params['Decompressed Evaluation Image Directory']],
    # repetition
    [params['Compressed Repetition Training Folder Directory'],
     params['Decompressed Repetition Training Folder Directory']]
])

# Get list of directories of training batches
train_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Train Image Directory'])
# Get list of directories of evaluation batches
eval_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Evaluation Image Directory'])
# Get list of directories of repetition batches
repit_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Repetition Training Folder Directory'])

if params['Debug Mode']:
    print 'train shape: ' + str(np.asarray(train_batches).shape)
    print 'evaluation shape: ' + str(np.asarray(eval_batches).shape)

# Creates the Neural Network
model = helper_functions.makeModel(params)

# Loads previous model if it exists and the option is chosen
if params['Should Load Model'] and isfile(params['Path To Model To Load']):
    model.load_weights(params['Path To Model To Load'])

# prints error if it should load model but the path to the model is false:
if params['Should Load Model'] and (isfile(params['Path To Model To Load']) is False):
    print 'Was told to load the model but the given path is false, so the model was not loaded.'

# Prints a summary of the model and plot a picture if debug mode is on
if params['Debug Mode']:
    print model.summary()
    plot_model(model, to_file=params['Model Image Filename'], show_shapes=True)

# Main Program Loop
# Clear Eval Output Folder
helper_functions.clearFolder(params['Evaluation Save Directory'])
# Make A Progress Bar
pbar = progressbar.ProgressBar()
# loops through the number of training batches
# only runs if the program should retrain the model
if params['Should Load Model'] is False:
    for trainBatchNum in pbar(range(len(train_batches))):
        print "Batch " + str(trainBatchNum)

        # Load Current Training Batch
        train_batch = helper_functions.loadBatch(train_batches[trainBatchNum])
        print('Training Model')
        # Train The Model On The Batch
        model.fit(train_batch, train_batch, epochs=1, batch_size=params['Micro Batch Size'])

        # ID Of Current Image To Be Saved
        outImgID = 0
        print('Evaluating')
        for evalBatchNum in range(len(eval_batches)):
            # Load Current Evaluation Batch
            eval_batch = helper_functions.loadBatch(eval_batches[evalBatchNum])
            # Generate Images On Batch
            evaluations = model.predict(eval_batch, batch_size=1)
            # Save Each Generated Image And Its Corresponding Input
            for i, evaluation in enumerate(evaluations):
                helper_functions.saveImg(eval_batch[i], params['Evaluation Save Directory'] + "/Batch " + str(
                    trainBatchNum) + "-Eval " + str(outImgID) + "-input.png", params)
                helper_functions.saveImg(evaluation, params['Evaluation Save Directory'] + "/Batch " + str(
                    trainBatchNum) + "-Eval " + str(outImgID) + "-output.png", params)
                outImgID += 1

        print('Saving Model')
        # saves the model
        model.save_weights(params['Path To Export Model To'])

# For loop that does the repetitions of the images in the repetition folder
outImgID = 0
# makes a new folder to save the repetition evaluation images
os.makedirs(params['Evaluation Save Directory'] + '/repetitions')

for r in range(params['Repetition Image Repetitions']):
    print 'Repetition: ' + str(r + 1) + '/' + str(params['Repetition Image Repetitions'])
    # loop that trains through the repetition images once
    for trainBatchNum in range(len(repit_batches)):

        # Load Current Training Batch
        train_batch = helper_functions.loadBatch(repit_batches[trainBatchNum])

        print('Training Model...')
        # Train The Model On The Batch
        model.fit(train_batch, train_batch, epochs=1, batch_size=params['Micro Batch Size'])

        # print('Saving Model...')
        # # saves the model
        # model.save_weights(params['Path To Export Model To'])

        # ID Of Current Image To Be Saved
        print('Evaluating...')
        for evalBatchNum in range(len(eval_batches)):
            # Load Current Evaluation Batch
            eval_batch = helper_functions.loadBatch(eval_batches[evalBatchNum])
            # Generate Images On Batch
            evaluations = model.predict(train_batch, batch_size=1)
            # Save Each Generated Image And Its Corresponding Input
            for i, evaluation in enumerate(evaluations):
                helper_functions.saveImg(train_batch[i],
                                         params['Evaluation Save Directory'] + "/repetitions/Batch " + str(
                                             trainBatchNum) + "-Eval " + str(outImgID) + "-input-repetition.png",
                                         params)

                helper_functions.saveImg(evaluation, params['Evaluation Save Directory'] + "/repetitions/Batch " + str(
                    trainBatchNum) + "-Eval " + str(outImgID) + "-output-repetition.png", params)

                outImgID += 1
