############################################################## 
# Licensed under the MIT License                             #
# Copyright (c) 2018 Stefan Hein Bengtson and Malte Pedersen #
# See the file LICENSE for more information                  #
##############################################################

import cv2, sys, os.path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib

### Module imports ###
sys.path.append('../')

class Triangulate:
    # Calculates the intersection between two rays
    # A ray is defined as a:
    # - direction vector (i.e. r1D and r2D)
    # - a point on the ray (i.e. r1P and r2P)
    # source: http://morroworks.com/Content/Docs/Rays%20closest%20point.pdf
    def rayIntersection(self, ray1Dir, ray1Point, ray2Dir, ray2Point):
        a = ray1Dir
        b = ray2Dir
        A = ray1Point
        B = ray2Point
        c = B-A
        
        ab = np.dot(a,b)
        aa = np.dot(a,a)
        bb = np.dot(b,b)
        ac = np.dot(a,c)
        bc = np.dot(b,c)
    
        denom = aa*bb - ab*ab
        tD = (-ab*bc + ac*bb)/denom
        tE = (ab*ac - bc*aa)/denom

        D = A + a*tD
        E = B + b*tE
        point = (D+E)/2
        dist = np.linalg.norm(D-E)
        return point,dist

    # Refracts an incoming ray in a interface
    # Paramters:
    # - 'rayDir' is the vector of the incoming ray
    # - 'planeNormal' is the plane normal of the refracting interface
    # - 'n1' is the refraction index of the medium the ray travels >FROM<
    # - 'n2' is the refractio index of the medium the ray travels >TO<
    def refractRay(self, rayDir, planeNormal, n1, n2, verbose=False):
        r = n1/n2
        normPlane = planeNormal/np.linalg.norm(planeNormal)
        normDir = rayDir/np.linalg.norm(rayDir)
        c1 = np.dot(-normPlane,normDir)
        c2 = np.sqrt(1.0-r**2 * (1.0-c1**2))
        refracted = r*rayDir+(r*c1-c2)*normPlane
        if(verbose):
            print("c1: {0}".format(c1))
            print("test: {0}".format(1.0-r**2 * (1.0-c1**2)))
            print("Incidence angle: " + str(np.rad2deg(np.arccos(c1))))
            print("Refraction angle: " + str(np.rad2deg(np.arccos(c2))))
        return refracted,c1,c2

    # Internal function - do not call directly
    # Triangulates point while accounting for refraction
    def _triangulateRefracted(self, p1, p2, cam1, cam2, verbose=False):
        # 1) Backprojects points into 3D ray
        ray1 = cam1.backprojectPoint(*p1)
        ray2 = cam2.backprojectPoint(*p2)
        if(verbose):
            print("Ray1 \n -dir: {0}\n -point: {1}".format(*ray1))
            print("Ray2 \n -dir: {0}\n -point: {1}".format(*ray2))

        # 2) Find plane intersection
        p1Intersect = cam1.plane.intersectionWithRay(*ray1, verbose=verbose)
        p2Intersect = cam2.plane.intersectionWithRay(*ray2, verbose=verbose)
        if(verbose):
            print("Ray1 intersection: {0}".format(p1Intersect))
            print("Ray2 intersection: {0}".format(p2Intersect))

        # 3) Refract the backprojected rays
        n1 = 1.0 # Refraction index for air
        n2 = 1.33 # Refraction index for water
        ref1,_,_ = self.refractRay(ray1[0],cam1.plane.normal,n1,n2)
        ref2,_,_ = self.refractRay(ray2[0],cam2.plane.normal,n1,n2)
        if(verbose):
            print("Refracted ray1: {0}".format(ref1))
            print("Refracted ray2: {0}".format(ref2))

        # 4) Triangulate points the refracted rays
        rayIntersection = self.rayIntersection(ref1, p1Intersect, ref2, p2Intersect)

        # Plot stuff if enabled
        if(verbose):
            # Refracted ray 1
            cam1Pos = cam1.getPosition()
            newRay1 = 200 * ray1[0]
            newRay1 += cam1Pos[0]
            x1 = [cam1Pos[0][0], newRay1[0]]    
            y1 = [cam1Pos[0][1], newRay1[1]]
            z1 = [cam1Pos[0][2], newRay1[2]]

            ref1 /= np.linalg.norm(ref1)
            ref1 *= 200
            ref1 += p1Intersect
            x1r = [p1Intersect[0], ref1[0]]    
            y1r = [p1Intersect[1], ref1[1]]
            z1r = [p1Intersect[2], ref1[2]]    

            # Refracted ray 2
            cam2Pos = cam2.getPosition()
            newRay2 = 200 * ray2[0]
            newRay2 += cam2Pos[0]
            x2 = [cam2Pos[0][0], newRay2[0]]    
            y2 = [cam2Pos[0][1], newRay2[1]]
            z2 = [cam2Pos[0][2], newRay2[2]]

            ref2 /= np.linalg.norm(ref2)
            ref2 *= 200
            ref2 += p2Intersect
            x2r = [p2Intersect[0], ref2[0]]    
            y2r = [p2Intersect[1], ref2[1]]
            z2r = [p2Intersect[2], ref2[2]]

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot(x1, y1, z1)
            ax.plot(x1r, y1r, z1r, 'yellow')
            ax.plot(x2, y2, z2, 'red')
            ax.plot(x2r, y2r, z2r, 'green')
            ax.scatter(*cam1Pos[0], c='black')
            ax.scatter(*cam2Pos[0], c='black')
            ax.scatter(*rayIntersection[0], c='black', marker='x', s=20)
            ax.auto_scale_xyz([0, 40], [0, 40], [0, 40])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.show()        
        return rayIntersection[0], rayIntersection[1]

        # Internal function - do not call directly
    # Triangulates point using OpenCV's function
    # does not account for refraction
    def _triangulateOpenCv(self, p1, p2, cam1, cam2, verbose=False):
        # 1) Undistort points
        p1 = cv2.undistortPoints(np.array([[p1]]), cam1.K, cam1.dist)
        p2 = cv2.undistortPoints(np.array([[p2]]), cam2.K, cam2.dist)
        if(verbose):
            print("Undistorted top point: " + str(p1))
            print("Undistorted side point: " + str(p2))     

        # 2) Triangulate points using camera projection matrices
        point = cv2.triangulatePoints(cam1.getExtrinsicMat(),
                                      cam2.getExtrinsicMat(),
                                      p1,p2)
        point /= point[3]
        return point[:3].flatten(), -1.0

    # Triangulate 3D point using 2D points from two cameras
    # Parameters:
    # - 'p1' is 2D coordinates from camera 1
    # - 'p2' is 2D coordinates from camera 2
    # - 'cam1' is 'Camera' object for camera 1
    # - 'cam2' is 'Camera' object for camera 2
    # - 'correctRefraction' defines whether refraction should be accounted for
    def triangulatePoint(self, p1, p2, cam1, cam2, correctRefraction=True, verbose=False):    
        if(verbose):
            print("\n\nPoint 1: {0}".format(p1))
            print("Point 2: {0}".format(p2))
        if(correctRefraction):
            point, dist = self._triangulateRefracted(p1, p2, cam1, cam2, verbose=verbose)
        else:
            point, dist = self._triangulateOpenCv(p1, p2, cam1, cam2, verbose=verbose)
        if(verbose):
            print("Triangulated point: {0} with distance: {1}".format(point, dist))    
        return point, dist
