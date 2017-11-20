import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import cv2
import matplotlib.pyplot as plt

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from statistics import mean

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# for hist compare
import glob
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True,
	help = "Path to the directory of images")

ap.add_argument("-v", "--video", required = True,
	help = "videoPath")

args = vars(ap.parse_args())

def getHist(img):
	hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	cv2.normalize(hist, hist).flatten()
	return hist

def matchHist(hist_target, hist_candidate):
    # ("Correlation", cv2.HISTCMP_CORREL),
	# ("Chi-Squared", cv2.HISTCMP_CHISQR),
	# ("Intersection", cv2.HISTCMP_INTERSECT), 
	# ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
    return cv2.compareHist(hist_target, hist_candidate, cv2.HISTCMP_CORREL)

def imresize(src, height):
    ratio = src.shape[0] * 1.0/height
    width = int(src.shape[1] * 1.0/ratio)
    return cv2.resize(src, (width, height))


sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
PATH_TO_CKPT = './model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#NUM_CLASSES = 2
NUM_CLASSES = 90

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

indexes = []
images = []


for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
    filename = imagePath[imagePath.rfind("\\") + 1:]
    image = cv2.imread(imagePath)
    images.append(image)
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist).flatten()
    indexes.append(hist)

if not len(images):
    print("Can't read image !!")
    exit()

    
cv2.namedWindow("target")
cv2.moveWindow("target", 100, 200)
cv2.imshow("target", images[0])
cv2.waitKey(1)

videoPath = args["video"]
print("videoPath : ", videoPath)

if videoPath == "0":
    print("---- run camera ----")
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) #WIDTH
    cap.set(4, 480) #HEIGHT
else:
    print("---- run video ----")
    cap = cv2.VideoCapture(videoPath)
    
    
#out = cv2.VideoWriter("./test/output.avi", -1, 20.0, (903,480))

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, image = cap.read()
            if(ret ==0):
                break

            image = imresize(image, 480)
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            start_time = time.time()
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            elapsed_time = time.time() - start_time
            #print('inference time cost: {}'.format(elapsed_time))

            # make 1d array
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            # find people class
            searchValue = 1
            peoples = np.where(classes == searchValue)[0]

            # find valide bouding box using threshold
            img_width = image.shape[1]
            img_height = image.shape[0]
            thresh = 0.6
            people_boxes = []
            for i in peoples:
                if (scores[i] < thresh):
                    continue

                ymin, xmin, ymax, xmax = boxes[i]
                people_box = [int(xmin*img_width), int(ymin*img_height), 
                                int(xmax*img_width), int(ymax*img_height)]
                people_boxes.append(people_box)

            # compare with finding people
            thresh_hist = 0.95
            for box in people_boxes:
                cropped = image[box[1]:box[3], box[0]:box[2]]
                cropped_hist = getHist(cropped)

                hist_result = []
                for hist in indexes:
                    result = matchHist(hist, cropped_hist)
                    print("Hist result : " , result)
                    hist_result.append(result)

                hist_average = sum(map(float, hist_result))/float(len(hist_result))
                print("Hist average Result : ", hist_average)
                
                if (hist_average > thresh_hist):
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 8)
                else:
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 128, 128), 1)

            # Visualization of the results of a detection.
            # vis_util.visualize_boxes_and_labels_on_image_array(
            #     image,
            #     np.squeeze(boxes),
            #     np.squeeze(classes).astype(np.int32),
            #     np.squeeze(scores),
            #     category_index,
            #     use_normalized_coordinates=True,
            #     line_thickness=4)
            
            cv2.imshow("detections", image)

            ch = cv2.waitKey(15)
            if ch == 27: #27 == ESC key
                break 
            if ch == 32: #32 == SPACE key
                while 1:
                    ch = cv2.waitKey(10)
                    if ch == 32 or ch == 27:
                        break

                if ch == 27:
                    break

            # if cv2.waitKey(15) != -1:
            #     break

            #out.write(image)

cap.release()
#out.release()
cv2.destroyAllWindows()