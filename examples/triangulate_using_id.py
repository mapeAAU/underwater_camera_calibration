############################################################## 
# Licensed under the MIT License                             #
# Copyright (c) 2018 Stefan Hein Bengtson and Malte Pedersen #
# See the file LICENSE for more information                  #
##############################################################

import argparse, sys, os.path
import numpy as np
import pandas as pd
from sklearn.externals import joblib

### Module imports ###
sys.path.append('../')
from source.Triangulate import Triangulate

# Description:
# Triangulates all possible 3D positions based on detections from 2 cameras
# Input: Path to data directory which should contain:
#        - 'detections.csv' = CSV file containing 2D detections from both cameras
#        - 'cam1/2.pkl' = pickle-files containing calibrated 'Camera' objects for both cameras
#        - 'cam1/2_references.json' = json files containing reference corners
#
# Output: CSV file containing the triangulated 3D positions
#         Output is saved to 'triangulated.csv' in the specified data directory
#         NOTE: All possible combinations of 2D detections are saved!
#
# Example of usage:
# > $python triangulate_using_id.py -f ../data/triangulate_using_id/detections.csv
#
#        -f is the path to the .csv-file with detections.
#        -o is the offset, which describes the amount of frames to shift recording 1 to ensure temporal alignment between the two recordings. 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser();
ap.add_argument("-f", "--csvPath", help="Path to the CSV file to triangulate");
ap.add_argument("-o", "--offset", help="Number of frames to shift camera 1");

args = vars(ap.parse_args());

# ARGUMENTS *************
if args.get("csvPath", None) is None:
    print('No path to the .csv file was provided. Try again!');
    sys.exit();
else:
    csvPath = args["csvPath"]
    path = os.path.dirname(os.path.abspath(csvPath))    

if args.get("offset", None) is None:
    print('No temporal offset was specified for camera 1. No offset will be used.');
    cam1Offset = 0
else:
    cam1Offset = int(args["offset"])

# Extract (x,y) points for a specific camera and frame number
# Return list of points (x,y)
def getPoints(dataframe,frame,camId):
    data = dataframe[dataframe["frame"]==frame]
    data = data[data["cam"]==camId]
    x = np.array((data.x),dtype=float)
    y = np.array((data.y),dtype=float)
    pts = []
    for i in range(len(x)):
        pts.append((x[i],y[i]))
    return pts

# 1) Check whether necessary files exists
path = os.path.dirname(os.path.realpath(csvPath))
cam1 = os.path.join(path,'cam1.pkl')
cam2 = os.path.join(path,'cam2.pkl')
if(not os.path.isfile(cam1)):
    print("Error finding camera calibration file: \n {0}".format(cam1))
    sys.exit(0)
if(not os.path.isfile(cam2)):
    print("Error finding camera calibration file: \n {0}".format(cam2))
    sys.exit(0)
    
cam1ref = os.path.join(path,'cam1_references.json')
cam2ref = os.path.join(path,'cam2_references.json')
if(not os.path.isfile(cam1ref)):
    print("Error finding camera corner reference file: \n {0}".format(cam1ref))
    sys.exit(0)
if(not os.path.isfile(cam2ref)):
    print("Error finding camera corner reference file: \n {0}".format(cam2ref))
    sys.exit(0)
    
if(not os.path.isfile(csvPath)):
    print("Error finding CSV: \n {0}".format(csvPath))
    sys.exit(0)

# 2) Prepare cameras
cam1 = joblib.load(cam1)
cam1.calcExtrinsicFromJson(cam1ref)
print("Camera 1:")
print(" - position: \n" + str(cam1.getPosition()))
print(" - rotation: \n" + str(cam1.getRotationMat()))
print("")

cam2 = joblib.load(cam2)
cam2.calcExtrinsicFromJson(cam2ref)
print("Camera 2:")
print(" - position: \n" + str(cam2.getPosition()))
print(" - rotation: \n" + str(cam2.getRotationMat()))
print("")

# 3) Triangulate points
points = pd.read_csv(csvPath)
maxFrames = len(points['frame'].unique())
tr = Triangulate()
df = pd.DataFrame(columns=['frame','err','err1','err2',
                           '3d_x','3d_y','3d_z',
                           'cam1_x','cam1_y',
                           'cam2_x','cam2_y',
                           'cam1_proj_x','cam1_proj_y',
                           'cam2_proj_x','cam2_proj_y']);

correctRefract = True # Set this to False if you do not want to take refraction into account
frames = points['frame'].unique()
for i in frames:
    print("Frame {0}/{1}".format(i, maxFrames))
    if(i > maxFrames):
        break

    cam1Data = points[points['frame'] == i+cam1Offset]
    cam1Data = cam1Data[cam1Data['cam'] == 1]
    
    cam2Data = points[points['frame'] == i]
    cam2Data = cam2Data[cam2Data['cam'] == 2]

    tracks = cam1Data['id'].unique()    

    pos3d = []
    errors = []
    errors1 = []
    errors2 = []
    cam1Proj = []
    cam2Proj = []
    cam1Pos = []
    cam2Pos = []
    trackIds = []

    for t in tracks:
        track1 = cam1Data[cam1Data['id'] == t]
        if(len(track1) == 0):
            continue
        pos1 = (float(track1.x.values[0]), float(track1.y.values[0]))
        
        track2 = cam2Data[cam2Data['id'] == t]

        for l,pt in track2.iterrows():
            pos2 = (float(pt.x), float(pt.y))            
            p,d = tr.triangulatePoint(pos1,
                                      pos2,
                                      cam1,
                                      cam2,
                                      correctRefraction=correctRefract)
            
            p1 = cam1.forwardprojectPoint(*p, correctRefraction=correctRefract)
            p2 = cam2.forwardprojectPoint(*p, correctRefraction=correctRefract)
            
            err1 = np.linalg.norm(p1-pos1)
            err2 = np.linalg.norm(p2-pos2)
            err = err1+err2

            pos3d.append(p)
            errors.append(err)
            errors1.append(err1)
            errors2.append(err2)
            cam1Proj.append(p1)
            cam2Proj.append(p2)
            cam1Pos.append(pos1)
            cam2Pos.append(pos2)
            trackIds.append(t)
            
    newFrame = pd.DataFrame({
        'frame':[i]*len(pos3d),
        'err':errors,
        'err1':errors1,
        'err2':errors2,
        'id':trackIds,
        '3d_x':[p[0] for p in pos3d],
        '3d_y':[p[1] for p in pos3d],
        '3d_z':[p[2] for p in pos3d],
        'cam1_x':[p[0] for p in cam1Pos],
        'cam1_y':[p[1] for p in cam1Pos],
        'cam2_x':[p[0] for p in cam2Pos],
        'cam2_y':[p[1] for p in cam2Pos],
        'cam1_proj_x':[p[0] for p in cam1Proj],
        'cam1_proj_y':[p[1] for p in cam1Proj],
        'cam2_proj_x':[p[0] for p in cam2Proj],
        'cam2_proj_y':[p[1] for p in cam2Proj]});
    df = df.append(newFrame, ignore_index=True)
outputPath = csvPath.replace('.csv', '_triangulated.csv')
print("Saving data to: {0}".format(outputPath))
df.to_csv(outputPath)
