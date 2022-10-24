## Calibration of the intrinsic parameters
---

To estimate the intrinsic parameters of the camera, run the file **calibrate_intrinsics.py**  
You can test the program on the images placed in ../data/checkerboard_images/  
The parameters will be saved as **camera.pkl** in the given image-folder.

Example of usage:

```$python calibrate_intrinsics.py -cs 9 6 -ss 0.935 -if ../data/checkerboard_images/ -it .jpg```
 
* -cs is the number of intersection points in the outer layer of squares on the checkerboard. In the given example the checkerboard has 9x6 intersection points in the outer layer.
* -ss is the size of the squares in centimeters
* -if is the image-folder
* -it is the image-type (.jpg, .png, etc.)

## Triangulate detections using ID
---

To triangulate all possible 3D positions based on detections from 2 cameras, run the
file *triangulate_using_id.py*

Input: Path to data directory which should contain:

* **detections.csv** is the CSV file containing 2D detections from both cameras
* **cam1.pkl** is the pickle-file containing the calibrated 'Camera' object for camera 1
* **cam2.pkl** is the pickle-file containing the calibrated 'Camera' object for camera 2
* **cam1_references.json** is the json file containing the reference corners of e.g. the aquarium seen from camera 1
* **cam2_references.json** is the json file containing the reference corners of e.g. the aquarium seen from camera 2
          
Output: A CSV file containing the triangulated 3D positions

* The output is saved to **triangulated.csv** in the specified data directory
* **NOTE**: All possible combinations of 2D detections are saved!

Example of usage:

```$python triangulate_using_id.py -f ../data/triangulate_using_id/detections.csv -o 0```

* -f is the path to the .csv-file with detections.
* -o is the offset, which describes the amount of frames to shift recording 1 to ensure temporal alignment between the two recordings. 


