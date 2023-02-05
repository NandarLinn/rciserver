import cv2
import numpy as np
import os
from fnmatch import filter
from natsort import natsorted
from PIL import Image
import glob
import argparse
import time
import math
import shutil

# labelsPath = "/home/gigabyte/workspace/darknet/data/obj.names"

def bbox2yolo(size, box):
    xmin = box[0]
    xmax = box[2]
    ymin = box[1]
    ymax = box[3]

    xcen = float((xmin + xmax)) / 2 / size[1]
    ycen = float((ymin + ymax)) / 2 / size[0]

    w = abs(float((xmax - xmin)) / size[1])
    h = abs(float((ymax - ymin)) / size[0])

    return xcen, ycen, w, h


def remove_smaller_bounding_boxes(bboxes):
    bb = sorted(bboxes, key=lambda b: b[1])
    bb = np.array(bb)
    x1 = bb[:, 0]
    y1 = bb[:, 1]
    x2 = bb[:, 2]
    y2 = bb[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    avg_area = sum(area) / len(area)
    bb = list(bb)
    idx = []
    for i, a in enumerate(area):
        if a < (avg_area * 0.3):
            idx.append(i)
    idx = sorted(idx, reverse=True)
    for i in idx:
        bb.pop(i)
    return bb


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlap_thresh):
    new_boxes = []
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and
    # sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # idxs = np.argsort(y2)
    idxs = np.argsort(area)
    # keep looping while some indexes still remain in the indexes
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]  # the index of biggest value in y2

        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        # print("XX1 result", xx1)
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        bb = idxs[np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
        new_boxes.append([min(x1[bb]), min(y1[bb]), max(x2[bb]), max(y2[bb])])

    # return only the bounding boxes that were picked using the integer data type
    return new_boxes  # boxes[pick].astype("int")


# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def get_outputs_names(model):
    layers_names = model.getLayerNames()
    return [layers_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]


def start_points(size, split_size, overlap=0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points        

def detect(img_path, weightsPath, configPath, j, i):
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    color: tuple = (0, 255, 0)
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # load our input image and grab its spatial dimensions
    image = cv2.imread(img_path)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    conf_thresh = 0.4
    nms_thresh = 0.7
    xpoint = j
    ypoint = i

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > conf_thresh:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)
    
    r_boxes = []
    class_ids = []
    conf_score = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0]+xpoint, boxes[i][1]+ypoint)
            (w, h) = (boxes[i][2], boxes[i][3])
            class_ids.append(classIDs[i])
            conf_score.append(confidences[i])
            r_boxes.append([x, y, x + w, y + h])
    yolo_boxes = ''
    bb= non_max_suppression_fast(np.array(r_boxes), 0.7)
    for i, b in enumerate(r_boxes):
        b = [int(c) for c in b]  # to int
        yolo_box = bbox2yolo(image.shape[:2], b)
        yolo_boxes += f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"

    return bb