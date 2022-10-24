############################################################## 
# Licensed under the MIT License                             #
# Copyright (c) 2018 Stefan Hein Bengtson and Malte Pedersen #
# See the file LICENSE for more information                  #
##############################################################

import numpy as np

class Plane:
    def __init__(self, points=None):
        self.points = None
        self.normal = None
        self.x = None
        self.y = None
        if(points is not None):
            self.normal = self.calculateNormal(points)
            self.points = points

    # Calculates the plane normal n = [a b c] and d
    # for the plane: ax + by + cz + d = 0
    def calculateNormal(self, points, verbose=False):
        if(len(points) < 4):
            print("Error calculating plane normal. 4 or more points needed")
        #Calculate plane normal
        self.x = points[1]-points[2]
        self.x = self.x/np.linalg.norm(self.x)
        self.y = points[3]-points[2]
        self.y = self.y/np.linalg.norm(self.y)
        n = np.cross(self.x,self.y)
        n /= np.linalg.norm(n)
        if(verbose):
            print("Plane normal: \n {0} \n plane d: {1}".format(n,d))
        return n

    # Calcuates the intersection between a plane and a ray
    # r = ray direction vector
    # r0 = point in the ray
    def intersectionWithRay(self, r, r0, verbose=False):
        n0 = self.points[0]
        t = np.dot((n0 - r0), self.normal)
        t /= np.dot(r,self.normal)
        intersection = (t * r) + r0
        if(verbose):
            print("t: \n" + str(t))
            print("Intersection: \n" + str(intersection))
        return intersection.flatten()
