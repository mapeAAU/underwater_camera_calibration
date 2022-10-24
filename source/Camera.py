############################################################## 
# Licensed under the MIT License                             #
# Copyright (c) 2018 Stefan Hein Bengtson and Malte Pedersen #
# See the file LICENSE for more information                  #
##############################################################

import cv2, glob, sys, re, json
import numpy as np

### Module imports ###
sys.path.append('../')
from source.Plane import Plane

class Camera:
    def __init__(self):
        self.dist = None  # Lens distortion coefficients
        self.K = None     # Intrinsic camera parameters
        self.R = None     # Extrinsic camera rotation
        self.t = None     # Extrinsic camera translation
        self.plane = None # Water interface
        self.roi = None   # Region of interest dictated by aquarium corners

    # Find intrinsic parameters for the camera using a folder of images
    def calibrateFromFolder(self, imageFolder, checkerboardSize, squareSize, verbose=False):
        imageNames = glob.glob(imageFolder)
        images = []
        if(verbose):
            print("Calibration image names:")
        for imgPath in imageNames:
            if(verbose):
                print(imgPath)
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            images.append(img)

        return self.calibrate(images, checkerboardSize, squareSize, verbose=verbose)
        
    # Find intrinsic parameters for the camera
    def calibrate(self, images, checkerboardSize, squareSize, verbose=False):
        if(len(images) < 1):
            print("Camera: Error - Too few images for calibration")
            return

        # Find checkerboard corners in each image
        objP = self.getObjectPoints(checkerboardSize, squareSize)
        objPoints = []
        imgPoints = []
        imgCounter = 0
        for img in images:
            ret, corners = cv2.findChessboardCorners(img, checkerboardSize, None)
            imgCounter += 1
            if(ret):
                objPoints.append(objP)
                imgPoints.append(corners)
            else:
                print("Camera: Info - Unable to find corners in an image during calibration")
            if(verbose):
                print("Camera calibration - progress: {0} / {1}".format(imgCounter,len(images)))

        # Calibrate the camera
        ret, intri, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img.shape[::-1], None, None, flags=cv2.CALIB_RATIONAL_MODEL)
        if(ret):
            self.dist = dist
            self.K = intri
            return intri, dist
        else:
            print("Camera: Error - Calibration failed!")
            return

    def getObjectPoints(self, checkerboardSize, squareSize):
        objP = np.zeros((checkerboardSize[0]*checkerboardSize[1],3), np.float32)
        objP[:,:2] = np.mgrid[0:checkerboardSize[0],0:checkerboardSize[1]].T.reshape(-1,2)*squareSize
        return objP

    # Calculate camera position
    # i.e. -R^-1 t
    def getPosition(self):
        if(self.R is None or self.t is None):
            print("Camera: Error - Extrinsic parameters is needed to find the camera postion")
            return
        rotMat = self.getRotationMat()
        camPos = -np.dot(rotMat.T, self.t)
        return camPos.T

    def getRotationMat(self):
        if(self.R is None):
            print("Camera: Error - Extrinsic parameters is needed to return rotation matrix")
            return
        return cv2.Rodrigues(self.R)[0]

    # Find extrinsic parameters for the camera using
    # image <--> world reference points from a CSV file
    def calcExtrinsicFromJson(self, jsonPath, method=None):
        # Load json file
        with open(jsonPath) as f:
            data = f.read()

        # Remove comments
        pattern = re.compile('/\*.*?\*/', re.DOTALL | re.MULTILINE)
        data = re.sub(pattern, ' ', data)
        
        # Parse json
        data = json.loads(data)

        # Convert to numpy arrays
        cameraPoints = np.zeros((4,1,2))
        worldPoints = np.zeros((4,3))

        for i,entry in enumerate(data):
            cameraPoints[i][0][0] = entry["camera"]["x"]
            cameraPoints[i][0][1] = entry["camera"]["y"]

            worldPoints[i][0] = entry["world"]["x"]
            worldPoints[i][1] = entry["world"]["y"]
            worldPoints[i][2] = entry["world"]["z"]

        # Calc extrinsic parameters
        if(method == None):
            self.calcExtrinsic(worldPoints.astype(float), cameraPoints.astype(float))
        else:
            self.calcExtrinsic(worldPoints.astype(float), cameraPoints.astype(float), method=method)

        self.rot = cv2.Rodrigues(self.R)[0]
        self.pos = self.getPosition()
    
    # Find extrinsic parameters for the camera
    # Mainly two methods:
    # cv2.SOLVEPNP_P3P and cv2.SOLVEPNP_ITERATIVE
    # See: http://docs.opencv.org/trunk/d9/d0c/group__calib3d.html#ggaf8729b87a4ca8e16b9b0e747de6af27da9f589872a7f7d687dc58294e01ea33a5
    def calcExtrinsic(self, worldPoints, cameraPoints, method=cv2.SOLVEPNP_ITERATIVE):
        if(self.K is None or self.dist is None):
            print("Camera: Error - Calibrate camera before finding extrinsic parameters!")
            return

        ret, rvec, tvec = cv2.solvePnP(worldPoints,cameraPoints,self.K,self.dist,flags=method)
        if(ret):
            self.R = rvec
            self.t = tvec
            self.plane = Plane(worldPoints)
            # Ensure that the plane normal points towards the camera
            if(np.dot(self.getPosition(), self.plane.normal) < 0):
                self.plane.normal = -self.plane.normal

            # Create roi
            roiPts = cv2.undistortPoints(cameraPoints, self.K, self.dist)
            roiPts = roiPts.reshape(4,2)
            self.roi = {}
            self.roi["x"] = (min(roiPts[:,0]), max(roiPts[:,0]))
            self.roi["y"] = (min(roiPts[:,1]), max(roiPts[:,1]))
            return rvec, tvec
        else:
            print("Camera: Error - Failed to find extrinsic parameters")
        return

    # Backproject 2D point into a 3D ray
    # i.e. finds R = R^-1 K^-1 [x y 1]^T
    def backprojectPoint(self, x, y):
        if(self.R is None or self.t is None):
            print("Camera: Error - Extrinsic parameters is needed to back-project a point")
            return
        if(self.K is None or self.dist is None):
            print("Camera: Error - Intrinsic parameters is needed to back-project a point")
            return
        
        # Calculate R = K^-1 [x y 1]^T and account for distortion
        ray = cv2.undistortPoints(np.array([[[x,y]]]), self.K, self.dist)
        ray = ray[0][0] # Unwrap point from array of array
        ray = np.array([ray[0], ray[1], 1.0])
        
        # Calculate R^-1 R
        ray = np.dot(np.linalg.inv(self.rot), ray)
        ray /= np.linalg.norm(ray)

        # Calculate camera center, i.e. -R^-1 t
        ray0 = self.pos
        return ray, ray0

    def forwardprojectPoint(self, x, y, z, correctRefraction=True, verbose=False):
        if(correctRefraction is False):
            p3 = cv2.projectPoints(np.array([[[x,y,z]]]), self.R, self.t, self.K, self.dist)[0]
            return p3.flatten()
        
        p1 = np.array([x,y,z])
        c1 = self.pos.flatten()
        w = self.plane.normal
        
        # 1) Plane between p1 and c1, perpendicular to w
        n = np.cross((p1-c1), w)
        if(verbose):
            print("Plane normal: {0}".format(n))

        # 2) Find plane origin and x/y directions
        #    i.e. project camera position onto refraction plane
        p0 = self.plane.intersectionWithRay(-w, c1)        
        if(verbose):
            print("Plane origin: {0}".format(p0))

        pX = c1-p0
        pX = pX / np.linalg.norm(pX)
        pY = np.cross(n, pX)
        pY = pY / np.linalg.norm(pY)
        if(verbose):
            print("Plane x direction: {0}".format(pX))
            print("Plane y direction: {0}".format(pY))
            print("Direction dot check: \n{0}\n{1}\n{2}".format(np.dot(pX,pY),
                                                                np.dot(n,pX),
                                                                np.dot(n,pY)))

        # 3) Project 3d position and camera position onto 2D plane
        p1_proj = np.array([np.dot(pX, p1-p0),
                            np.dot(pY, p1-p0)])
        c1_proj = np.array([np.dot(pX, c1-p0),
                            np.dot(pY, c1-p0)])
        if(verbose):
            print("P1 projection: {0}".format(p1_proj)) 
            print("C1 projection: {0}".format(c1_proj))

        # 4) Construct 4'th order polynomial
        sx = p1_proj[0]
        sy = p1_proj[1]
        e = c1_proj[0]
        r = 1.33
        N = (1/r**2) - 1

        y4 = N
        y3 = -2*N*sy
        y2 = (N * sy**2+(sx**2/r**2)-e**2)
        y1 = 2 * e**2 * sy
        y0 = -e**2 * sy**2

        coeffs = [y4, y3, y2, y1, y0]
        res = np.roots(coeffs)
        
        real = np.real(res)
        resRange = (min(1e-6,sy),max(1e-6,sy))

        finalRes = []
        for r in real:
            if(r > resRange[0] and r < resRange[1]):                
                finalRes.append(r)
        finalRes = finalRes[np.argmax([abs(x) for x in finalRes])]
        refPoint = (finalRes*pY)+p0
            
        if(verbose):
            print("\n")
            print("4th order poly details:")
            print(" - Range: {0}".format(resRange))
            print(" - Roots: {0}".format(real))
            print(" - finalRes: {0}".format(finalRes))
            print(" - pY: {0}".format(pY))
            print(" - p0: {0}".format(p0))
            print(" - Intersection point: {0}".format(refPoint))

        p3 = cv2.projectPoints(np.array([[[*refPoint]]]), self.R, self.t, self.K, self.dist)[0]
        return p3.flatten() 
