# Find People

## Object detecting

* CNN Model : mobilenet + ssd 
* training data : mscoco
    * need download to models directory
        * http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz
* development framework : tensorflow(v1.4)
* performance : run 30fps on GTX1070 GPU (no optimization)


## Image matching

* algorithm test
    * feature extraction
        * GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence
            * fail
            * feature extraction is suitable for high resolution image

* Histogram comparing
    * Correlation ( used )
    * Chi-Square
    * Intersection 
    * Bhattacharyya distance
* development library : opencv


## License
The code is based on GOOGLE tensorflow object detection api. Please refer to the license of tensorflow.

