## Interpretation of the data files
---

detections.csv
---
Before it is possible to calculate the 3D position of a given object, the object must have been detected in two displaced cameras simultaneously. In this example, the detections are placed in the file **detections.csv** and the layout of the file is shown below:

cam |frame |id |x  |y  |
----|------|---|---|---|
1   |2     |1  |100|353|
2   |2     |1  |430|512|

* **cam** is the ID of the camera from where the given detection is found
* **frame** is the frame number
* **id** is the ID of the given object
* **x** is the 2D image x-coordinate of the detection
* **y** is the 2D image y-coordinate of the detection


detections_triangulated.csv
---
After the detections have been processed in order to calculate the triangulated 3D positions of the given object, the results are placed in the **detections_triangulated.csv** file. The layout of the .csv-file is presented below. It should be noted that if only the x-coordinate of a given parameter for a single camera is explained, the same is applicable for the y- and z-coordinates and for both cameras.

3d_x | 3d_y | 3d_z | cam1_proj_x | cam1_proj_y | cam1_x | cam1_y | cam2_proj_x | cam2_proj_y | cam2_x | cam2_y| err | err1 | err2 | frame | id |
-----|------|------|-------------|-------------|--------|--------|-------------|-------------|--------|-------|-----|------|------|-------|----|
18.89| 9.89 | 18.27| 1237.59	 | 770.98      | 1237   | 771    | 1389.76     | 1139.00     | 1389	  | 1139  | 1.36| 0.59 | 0.76 | 1     | 0  |
22.76| 10.18| 18.44| 1096.79     | 770.98	   | 1097   | 771    | 1570.73     | 1149.01     | 1571	  | 1149  | 0.47| 0.20 | 0.26 | 1     | 1  |


* **3d_x** is the triangulated 3D x-coordinate of the detection with respect to the reference frame (e.g. outlined by the aquarium)
* **cam1_proj_x** is the 2D x-coordinate of the estimated 3D position after it has been reprojected back into the image plane of camera 1
* **cam1_x** is the original 2D image x-coordinate of the detection
* **err** is the total reprojection error given by: err = err1 + err2
* **err1** is the distance between the original 2D image position and the 2D position of the triangulated position after it has been reprojected back into the image plane of camera 1 
* **err2** is the same as **err1**, except for the reprojection being onto the image plane of camera 2
* **frame** is the frame number
* **id** is the ID of the given object
