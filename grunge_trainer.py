# Import Libraries
from os.path import isfile
import os
import numpy as np
import keras
from keras.models import load_model
from keras.utils import plot_model
import helper_functions
import progressbar



# Initialize Parameters for custumization
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
    'Layer Dimensions': [params['Image Dimensions']['x'] * params['Image Dimensions']['y'], 7000, 3000, 900, 3000, 7000, params['Image Dimensions']['x'] * params['Image Dimensions']['y']],
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

# uses the fucntion make_batches in the helper_functions folder to convert the images to h5 files in the batch sizes
# (also does it for the evaluation images)
helper_functions.make_batches(params, [
    # training
    [params['Compressed Train Image Directory'],
        params['Decompressed Train Image Directory']],
    # evaluation
    [params['Compressed Evaluation Image Directory'],
        params['Decompressed Evaluation Image Directory']],
    # repition
    [params['Compressed Repition Training Folder Directory'],
        params['Decompressed Repition Training Folder Directory']]
])

# Get list of directorys of training batches
train_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Train Image Directory'])
# Get list of directorys of evaluation batches
eval_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Evaluation Image Directory'])
# Get list of directorys of repition batches
repit_batches = helper_functions.loadH5sFromFolder(
    params['Decompressed Repition Training Folder Directory'])


if params['Debug Mode']:
    print 'train shape: ' + str(np.asarray(train_batches).shape)
    print 'evaluation shape: ' + str(np.asarray(eval_batches).shape)

'''
# tells user if they did not supply enough images for the training
if len(train_batches) < params['Batch Number']:
    print('BATCH ERROR:')
    print('I was told to run through ' + str(params['Batch Number']) + ' batches, yet I can only create ' + str(len(train_batches)) +
          ' batches with the number of images supplied. I need ' + str((params['Batch Number'] - len(train_batches)) * params['Batch Size']) + ' more images')

# tells user if program has enough evaluation images
if len(eval_batches) < params['Batch Number']:
    print('EVALUATION ERROR:')
    print('I was told to run through ' + str(params['Batch Number']) + ' batches, yet I can only create ' + str(len(eval_batches)) +
          ' evaluation sets with the number of images supplied. I need ' + str((params['Batch Number'] - len(eval_batches)) * params['Number Of Images To Evaluate On']) + ' more images')
'''
# Creates the Neural Network
model = helper_functions.makeModel(params)

# Loads previous model if it exists and the option is chosen
if params['Should Load Model'] and isfile(params['Path To Model To Load']):
    model.load_weights(params['Path To Model To Load'])

# prints error if it should load model but the path to the model is false:
if params['Should Load Model'] and (isfile(params['Path To Model To Load']) is False):
    print 'Was told to load teh model but the path givin is false, so the model was not loaded.'

# Prints a summary of the model and plot a picture if debug mode is on
if params['Debug Mode']:
    print model.summary()
    plot_model(model, to_file=params['Model Image Filename'], show_shapes=True)

# Main Program Loop
#Clear Eval Output Folder
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
            # Save Each Generated Imadsssge And Its Corresponding Input
            for i, evaluation in enumerate(evaluations):
                helper_functions.saveImg(eval_batch[i], params['Evaluation Save Directory'] + "/Batch " + str(
                    trainBatchNum) + "-Eval " + str(outImgID) + "-input.png", params)
                helper_functions.saveImg(evaluation, params['Evaluation Save Directory'] + "/Batch " + str(
                    trainBatchNum) + "-Eval " + str(outImgID) + "-output.png", params)
                outImgID += 1

        print('Saving Model')
        # saves the model
        model.save_weights(params['Path To Export Model To'])


# For loop that does the repitions of the images in the repition folder
outImgID = 0;
# makes a new folder to save the repition evaluation images
os.makedirs(params['Evaluation Save Directory']+'/repitions')

for r in range(params['Repetition Image Repitions']):
    print 'Repetition: '+str(r+1)+'/'+str(params['Repetition Image Repitions'])
    # loop that trains through the repition images once
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

                helper_functions.saveImg(train_batch[i], params['Evaluation Save Directory'] + "/repitions/Batch " + str(trainBatchNum) + "-Eval " + str(outImgID) + "-input-repition.png", params)

                helper_functions.saveImg(evaluation, params['Evaluation Save Directory'] + "/repitions/Batch " + str(trainBatchNum) + "-Eval " + str(outImgID) + "-output-repition.png", params)

                outImgID += 1
