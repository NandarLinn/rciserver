import os
import time
from datetime import datetime
from flask_cors import CORS, cross_origin
import cv2
import mysql.connector
import numpy as np
from PIL import Image
from flask import Flask
from flask import render_template, json, request, send_from_directory
# from gevent.pywsgi import WSGIServer
from app import app
import io
import base64

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.jpeg',
                               mimetype='image/vnd.microsoft.icon')

origin = "http://54.254.43.49:5000"
# origin = "http://10.10.111.24:5000"

app = Flask(__name__)
cors = CORS(app, resources={r"/foo": {"origins": origin}})

mysql_config = {
    'user': 'projectx',
    'password': 'H0meAl0ne!',
    'host': '127.0.0.1',
    'database': 'rebarnext',
    'raise_on_warnings': True
}

configPath = "utils/yolov4-tiny.cfg"
weightsPath = "utils/large_rebar_final.weights"
smallweight = "utils/yolov4-tiny_last.weights"

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
color: tuple = (0, 255, 0)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

smallnet = cv2.dnn.readNetFromDarknet(configPath, smallweight)
color: tuple = (0, 255, 0)

# determine only the *output* layer names that we need from YOLO
smallln = smallnet.getLayerNames()
smallln = [smallln[i[0] - 1] for i in smallnet.getUnconnectedOutLayers()]

# Calculate start point of each bounding box after splitting
def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size*(1-overlap))
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
        if a < (avg_area * 0.4):
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

def smalldetect(img_path, j, i):
    # load our input image and grab its spatial dimensions
    image = cv2.imread(img_path)
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    smallnet.setInput(blob)
    start = time.time()
    layerOutputs = smallnet.forward(smallln)
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

def detect(img_path, j, i):
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

@app.route('/')
def index():
    return render_template('index.html')

#Define image
@app.route('/countrebar', methods=["GET","POST"])
def countrebar():
    file = request.files['image']
    image=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    img = image.copy()
    split_ratio = 2
    overlap_ratio = 0.3
    img_h, img_w, _ = image.shape
    split_width = int(img_w/split_ratio)
    split_height = int(img_h/split_ratio)
    X_points = start_points(img_w, split_width, overlap_ratio)
    Y_points = start_points(img_h, split_height, overlap_ratio)
    color: tuple = (0, 255, 0)
    frmt = 'jpg'
    res_bb = []
    res = []
    yolo_boxes = ''
    for i in Y_points:
        for j in X_points:
            split = image[i:i+split_height, j:j+split_width]
            cv2.imwrite(f"tempic.jpg", split)
            rboxes = smalldetect(f"tempic.jpg", j, i)
            # print("Length of r boxes", rboxes)
            for bb in rboxes:
                res_bb.append(bb)
    bb = non_max_suppression_fast(np.array(res_bb), 0.7)
    bb = remove_smaller_bounding_boxes(bb)

    for b in bb:
        b = [int(c) for c in b]
        x1, y1, x2, y2 = b
        center_coordinates = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(min((x2 - x1) / 2, (y2 - y1) / 2))
        cv2.circle(img, center_coordinates, radius, color, 3)
        # cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), color, 5)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read()).decode("utf-8")
    res.append(len(bb))
    res.append(img_base64)

    return render_template("countbars.html", data=res)

@app.route('/greatercounterbarapi', methods=["GET","POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def greatercounterbarapi():
    start_time = time.time()
    file = request.files['image']
    image=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    filename = str(file.filename)
    filename = filename.split('.')[0]

    img = image.copy()
    t1 = time.time()
    split_ratio = 2
    overlap_ratio = 0.3
    img_h, img_w, _ = image.shape
    split_width = int(img_w/split_ratio)
    split_height = int(img_h/split_ratio)
    X_points = start_points(img_w, split_width, overlap_ratio)
    Y_points = start_points(img_h, split_height, overlap_ratio)
    color: tuple = (0, 255, 0)
    frmt = 'jpg'
    res_bb = []
    yolo_boxes = ''
    for i in Y_points:
        for j in X_points:
            split = image[i:i+split_height, j:j+split_width]
            cv2.imwrite(f"tempic.jpg", split)
            rboxes = detect(f"tempic.jpg", j, i)
            # print("Length of r boxes", len(rboxes))
            for bb in rboxes:
                res_bb.append(bb)
    bb = non_max_suppression_fast(np.array(res_bb), 0.7)
    print("before remove smallerboxes", len(bb))
    if len(bb)>0:
    	bb = remove_smaller_bounding_boxes(bb)
    print("after remove smallerboxes", len(bb))

    for b in bb:
        b = [int(c) for c in b]
        x1, y1, x2, y2 = b
        center_coordinates = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(min((x2 - x1) / 2, (y2 - y1) / 2))
        cv2.circle(img, center_coordinates, radius, color, 3)
        # cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), color, 5)

    for a, b in enumerate(bb):
        b = [int(c) for c in b]
        yolo_box = bbox2yolo(image.shape[:2], b)
        yolo_boxes += f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"

    cv2.imwrite(f"log/{filename}.jpg", image)
    with open(f"log/{filename}.txt", 'w+') as file:
        file.write(yolo_boxes)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image.astype("uint8"))
    img.save(f"output_img/{filename}.jpg")
    # img_url = f"http://0.0.0.0:5000/output_img/{filename}.jpg"
    img_url = f"http://54.254.43.49/output_img/{filename}.jpg"
    result = {'counts': len(bb), 'img': img_url}
    # rawBytes = io.BytesIO()
    # img.save(rawBytes, "JPEG")
    # rawBytes.seek(0)
    # img_base64 = base64.b64encode(rawBytes.read()).decode("utf-8")

    # result = {'counts': len(bb), 'img': img_base64}
    return json.dumps(result)
    #
@app.route('/smallerbarapi', methods=["GET","POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def counterbarapi():
   start_time = time.time()
   file = request.files['image']
   image=cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
   if image.shape[2] == 4:
       image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

   filename = str(file.filename)
   filename = filename.split('.')[0]

   img = image.copy()
   t1 = time.time()
   split_ratio = 2
   overlap_ratio = 0.3
   img_h, img_w, _ = image.shape
   split_width = int(img_w/split_ratio)
   split_height = int(img_h/split_ratio)
   X_points = start_points(img_w, split_width, overlap_ratio)
   Y_points = start_points(img_h, split_height, overlap_ratio)
   color: tuple = (0, 255, 0)
   frmt = 'jpg'
   res_bb = []
   yolo_boxes = ''
   for i in Y_points:
       for j in X_points:
           split = image[i:i+split_height, j:j+split_width]
           cv2.imwrite(f"tempic.jpg", split)
           rboxes = smalldetect(f"tempic.jpg", j, i)
           # print("Length of r boxes", len(rboxes))
           for bb in rboxes:
               res_bb.append(bb)
   bb = non_max_suppression_fast(np.array(res_bb), 0.7)
   print("before removing smaller boxes", len(bb))
   if len(bb)>0:
   		bb = remove_smaller_bounding_boxes(bb)
   print("after removing smaller boxes",len(bb))

   for b in bb:
       b = [int(c) for c in b]
       x1, y1, x2, y2 = b
       center_coordinates = (int((x1 + x2) / 2), int((y1 + y2) / 2))
       radius = int(min((x2 - x1) / 2, (y2 - y1) / 2))
       cv2.circle(img, center_coordinates, radius, color, 3)
       # cv2.rectangle(img, (b[0],b[1]), (b[2],b[3]), color, 5)

   for a, b in enumerate(bb):
       b = [int(c) for c in b]
       yolo_box = bbox2yolo(image.shape[:2], b)
       yolo_boxes += f"0 {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}\n"

   cv2.imwrite(f"log/{filename}.jpg", image)
   with open(f"log/{filename}.txt", 'w+') as file:
       file.write(yolo_boxes)

   image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   img = Image.fromarray(image.astype("uint8"))
   img.save(f"output_img/{filename}.jpg")
   img.save(f"output_img/{filename}.jpg")
   # img_url = f"http://0.0.0.0:5000/output_img/{filename}.jpg"
   img_url = f"http://54.254.43.49/output_img/{filename}.jpg"
   result = {'counts': len(bb), 'img': img_url}

   return json.dumps(result)
   
@app.route('/exportdata', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def exportdata():
    data = request.json
    customer_name = data['customer_name']
    receipt_date = data['receipt_date']
    truck_driver = data['truck_driver']
    truck_plate_number = data['truck_plate_number']
    warehouse_location = data['warehouse_location']
    destination = data['destination']
    sale_contact_number = data['sale_contact_number']
    receiver = data['receiver']
    warehouse_authorized = data['warehouse_authorized']
    user_id = data['user']
    bundle_tag_list = data['bundle_tag_list']

    if receipt_date:
        datetime_obj = datetime.strptime(receipt_date.replace(',', ''), '%b %d %Y')
    else:
        datetime_obj = datetime.strptime("Sep 9 9999", '%b %d %Y')

    print(customer_name, sale_contact_number, truck_driver, truck_plate_number, warehouse_authorized,
          destination, warehouse_location, receiver, user_id)

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    add_receipt_row = ("INSERT INTO receipt"
                       "(cust_name, receipt_date, truck_driver, truck_plate_no,"
                       "warehouse_location, destination, sale_contact_no, receiver,"
                       "warehouse_authorized, user_id, timestamp) "
                       "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())")
    data_receipt = (
        customer_name, datetime_obj.date(), truck_driver, truck_plate_number, warehouse_location, destination,
        sale_contact_number, receiver, warehouse_authorized, user_id)

    cursor.execute(add_receipt_row, data_receipt)
    cnx.commit()

    for row in bundle_tag_list:
        print(row[0], row[1], row[2], row[3], row[4])
        manufacturer = row[0]
        grade = row[1]
        size = row[2]
        manufactured_date = row[3]
        count = row[4]
        ai_count = row[5]

        add_bundletag_row = ("INSERT INTO bundletag"
                             "(manufacturer, grade, size, manufactured_date, count, ai_count, receipt_id)"
                             "VALUES (%s, %s, %s, %s, %s, %s, (SELECT MAX(id) FROM rebarnext.receipt))")
        data_bundletag = (manufacturer, grade, size, manufactured_date, count, ai_count)
        cursor.execute(add_bundletag_row, data_bundletag)
        cnx.commit()

    select_receipt_id = "SELECT MAX(id) FROM receipt"
    cursor.execute(select_receipt_id)
    query = cursor.fetchall()
    cnx.commit()
    cursor.close()

    return json.dumps({"receipt_id": query[0][0]})


@app.route('/adduser', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def adduser():
    data = request.json
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    add_user = "INSERT INTO rebarnext.user (email) VALUES (%(email)s);"
    data_user = {"email": user_email}
    cursor.execute(add_user, data_user)
    cnx.commit()
    cursor.close()
    return "New user added"


@app.route('/addrollmark', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def addrollmark():
    data = request.json
    rollmark = data['rollmark']
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    add_rollmark = "INSERT INTO rebarnext.rebarmanufacturer (manufacturer, user) VALUES (%(manufacturer)s, %(email)s);"
    data_rollmark = {"manufacturer": rollmark, "email": user_email}
    cursor.execute(add_rollmark, data_rollmark)
    cnx.commit()
    cursor.close()
    return "New rollmark added"


@app.route('/addgrade', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def addgrade():
    data = request.json
    grade = data['grade']
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    add_grade = "INSERT INTO rebarnext.rebargrade (grade, user) VALUES (%(grade)s, %(email)s);"
    data_grade = {"grade": grade, "email": user_email}
    cursor.execute(add_grade, data_grade)
    cnx.commit()
    cursor.close()
    return "New grade added"


@app.route('/addwarehouse', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def addwarehouse():
    data = request.json
    warehouse = data['warehouse']
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    add_warehouse = "INSERT INTO rebarnext.rebarwarehouse (warehouse, user) VALUES (%(wh)s, %(email)s);"
    data_warehouse = {"wh": warehouse, "email": user_email}
    cursor.execute(add_warehouse, data_warehouse)
    cnx.commit()
    cursor.close()
    return "New warehouse added"


@app.route('/getrollmarklist', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def getrollmarklist():
    data = request.json
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    select_rollmark_row = "SELECT manufacturer, user FROM rebarmanufacturer WHERE (user=%(email)s OR user is null)"
    data_rollmark = {'email': user_email}
    cursor.execute(select_rollmark_row, data_rollmark)
    rows = cursor.fetchall()

    rollmark_list = []
    for eachRow in rows:
        rollmark_list.append([str(eachRow[0]), str(eachRow[1])])
    cnx.commit()
    cursor.close()
    return json.dumps({"rollmark_list": rollmark_list})


@app.route('/getgradelist', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def getgradelist():
    data = request.json
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    select_grade_row = "SELECT grade, user FROM rebargrade WHERE (user=%(email)s OR user is null)"
    data_grade = {'email': user_email}
    cursor.execute(select_grade_row, data_grade)
    rows = cursor.fetchall()

    grade_list = []
    for eachRow in rows:
        grade_list.append([str(eachRow[0]), str(eachRow[1])])
    cnx.commit()
    cursor.close()
    return json.dumps({"grade_list": grade_list})


@app.route('/getwarehouselist', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def getwarehouselist():
    data = request.json
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    select_warehouse_row = "SELECT warehouse, user FROM rebarwarehouse WHERE (user=%(email)s OR user is null)"
    data_warehouse = {'email': user_email}
    cursor.execute(select_warehouse_row, data_warehouse)
    rows = cursor.fetchall()

    warehouse_list = []
    for eachRow in rows:
        warehouse_list.append([str(eachRow[0]), str(eachRow[1])])
    cnx.commit()
    cursor.close()
    return json.dumps({"warehouse_list": warehouse_list})


@app.route('/getreceiptlist', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def getreceiptlist():
    data = request.json
    user_email = data['user_email']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    select_receipt_row = "SELECT id, receipt_date FROM receipt WHERE user_id=%(email)s"
    data_receipt = {'email': user_email}
    cursor.execute(select_receipt_row, data_receipt)
    rows = cursor.fetchall()

    receipt_list = []
    for eachRow in rows:
        receipt_list.append([str(eachRow[0]), str(eachRow[1])])
    cnx.commit()
    cursor.close()
    return json.dumps({"receipt_list": receipt_list})


@app.route('/getreceiptdata', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def getreceiptdata():
    data = request.json
    receipt_id = data['receipt_id']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    select_receipt_row = "SELECT cust_name, receipt_date, truck_driver, truck_plate_no," \
                         "warehouse_location, destination, sale_contact_no, receiver, warehouse_authorized " \
                         "FROM receipt " \
                         "WHERE id=%(id)s"
    select_bundletag_data = "SELECT manufacturer, grade, size, manufactured_date, count FROM rebarnext.bundletag " \
                            "JOIN rebarnext.receipt ON rebarnext.receipt.id = rebarnext.bundletag.receipt_id " \
                            "WHERE rebarnext.receipt.id=%(id)s;"
    data_receipt = {'id': receipt_id}
    cursor.execute(select_receipt_row, data_receipt)

    receipt_row = cursor.fetchall()
    cursor.execute(select_bundletag_data, data_receipt)
    receipt_list = []
    for r in receipt_row:
        receipt_list.append([str(r[0]), str(r[1]), r[2], r[3], r[4], r[5], r[6], r[7], r[8]])

    bundletag_data = cursor.fetchall()
    cnx.commit()
    cursor.close()
    return json.dumps({"receipt_row": receipt_list, "bundletag_data": bundletag_data})


@app.route('/removereceiptrecord', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def removereceiptrecord():
    data = request.json
    receipt_id = data['receipt_id']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()

    move_receipt_row = "INSERT INTO receipt_backup " \
                       "(cust_name, receipt_date, truck_driver, truck_plate_no," \
                       "warehouse_location, destination, sale_contact_no, receiver," \
                       "warehouse_authorized, user_id, timestamp)" \
                       "(SELECT cust_name, receipt_date, truck_driver, truck_plate_no," \
                       "warehouse_location, destination, sale_contact_no, receiver," \
                       "warehouse_authorized, user_id, timestamp FROM receipt WHERE id=%(id)s)"

    remove_receipt_row = "DELETE FROM receipt WHERE id=%(id)s"
    form_id = {'id': receipt_id}
    cursor.execute(move_receipt_row, form_id)
    cursor.execute(remove_receipt_row, form_id)
    cnx.commit()
    cursor.close()
    return "Record of the receipt table deleted!"


@app.route('/removerollmark', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def removerollmark():
    data = request.json
    rollmark = data['rollmark']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_rollmark_row = "DELETE FROM rebarmanufacturer WHERE manufacturer=%(rm)s"
    rollmark = {'rm': rollmark}
    cursor.execute(remove_rollmark_row, rollmark)
    cnx.commit()
    cursor.close()
    return "Rollmark deleted!"


@app.route('/removegrade', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def removegrade():
    data = request.json
    grade = data['grade']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_grade_row = "DELETE FROM rebargrade WHERE grade=%(gr)s"
    data_grade = {'gr': grade}
    cursor.execute(remove_grade_row, data_grade)
    cnx.commit()
    cursor.close()
    return "Grade deleted!"


@app.route('/removewarehouse', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def removewarehouse():
    data = request.json
    warehouse = data['warehouse']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_warehouse_row = "DELETE FROM rebarwarehouse WHERE warehouse=%(wh)s"
    data_warehouse = {'wh': warehouse}
    cursor.execute(remove_warehouse_row, data_warehouse)
    cnx.commit()
    cursor.close()
    return "Warehouse deleted!"


@app.route('/editrollmark', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def editrollmark():
    data = request.json
    old_rollmark = data['old_rollmark']
    new_rollmark = data['new_rollmark']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_rollmark_row = "UPDATE rebarmanufacturer SET manufacturer=%(new)s WHERE manufacturer=%(old)s"
    rollmark = {'old': old_rollmark, 'new': new_rollmark}
    cursor.execute(remove_rollmark_row, rollmark)
    cnx.commit()
    cursor.close()
    return "Rollmark modified!"


@app.route('/editgrade', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def editgrade():
    data = request.json
    old_grade = data['old_grade']
    new_grade = data['new_grade']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_rollmark_row = "UPDATE rebargrade SET grade=%(new)s WHERE grade=%(old)s"
    rollmark = {'old': old_grade, 'new': new_grade}
    cursor.execute(remove_rollmark_row, rollmark)
    cnx.commit()
    cursor.close()
    return "Grade modified!"


@app.route('/editwarehouse', methods=["POST"])
@cross_origin(origin=origin, headers=['Content- Type', 'Authorization'])
def editwarehouse():
    data = request.json
    old_warehouse = data['old_warehouse']
    new_warehouse = data['new_warehouse']

    cnx = mysql.connector.connect(**mysql_config)
    cursor = cnx.cursor()
    remove_warehouse_row = "UPDATE rebarwarehouse SET warehouse=%(new)s WHERE warehouse=%(old)s"
    data_warehouse = {'old': old_warehouse, 'new': new_warehouse}
    cursor.execute(remove_warehouse_row, data_warehouse)
    cnx.commit()
    cursor.close()
    return "Warehouse modified!"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
