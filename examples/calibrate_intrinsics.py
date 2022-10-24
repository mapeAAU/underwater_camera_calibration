############################################################## 
# Licensed under the MIT License                             #
# Copyright (c) 2018 Stefan Hein Bengtson and Malte Pedersen #
# See the file LICENSE for more information                  #
##############################################################

import argparse
import os
import sys
from sklearn.externals import joblib

sys.path.append('../');
from source.Camera import Camera

# Description:
# Calibrates a camera using the images in the specified folder
# The parameters are saved in 'camera.pkl' in the same folder
#
# Example of usage:
# > $python calibrate_intrinsics.py -cs 9 6 -ss 0.935 -if ../data/checkerboard_images/ -it .jpg
#
#       -cs is the number of squares on the checkerboard. In the given example the checkerboard has 9x6 squares.
#       -ss is the size of the squares in centimeters
#       -if is the image-folder
#       -it is the image-type (.jpg, .png, etc.) 

ap = argparse.ArgumentParser();
ap.add_argument('-cs', '--checkerboardSize', nargs='+', type=int, help='number of squares on the checkerboard. E.g. [9 6] for a checkerboard that is 9 squares long and 6 squares wide.');
ap.add_argument('-ss', '--squareSize', help='size of the squares in centimeters');
ap.add_argument('-if', '--imageFolder', help='path to the folder containing the calibration images');
ap.add_argument('-it', '--imageType', help='type of image files. E.g. .png or .jpg');

args = vars(ap.parse_args());

if args.get('checkerboardSize', None) is None:
    print('Please specify checkerboard size');
    sys.exit();
else:
    checkerboardSize = tuple(args['checkerboardSize'])

if args.get('squareSize', None) is None:
    print('Please specify square size');
    sys.exit();
else:
    squareSize= float(args['squareSize'])

if args.get('imageFolder', None) is None:
    print('Please specify the path to the image folder');
    sys.exit();
else:
    imageFolder = args['imageFolder']

if args.get('imageType', None) is None:
    print('Please specify type of image (.jpg or .png)');
    sys.exit();
else:
    imageType = args['imageType']

print('Calibrating camera using the following parameters:')
print(' - image folder: ' + imageFolder)
print(' - checkerboard size: ' + str(checkerboardSize))
print(' - square size. ' + str(squareSize))
      
cam = Camera()
cam.calibrateFromFolder(imageFolder + '*' + imageType, checkerboardSize, squareSize, verbose=True)

print('Intrinsic: \n' + str(cam.K))
print('Distortion: \n' + str(cam.dist))

outputName = os.path.join(imageFolder, 'camera.pkl')
print('Saving camera to: ' + str(outputName))
joblib.dump(cam, outputName)
print('Done!')
