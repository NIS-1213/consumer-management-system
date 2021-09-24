import cv2
import datetime
import imutils
import numpy as np
from centroidtracker import CentroidTracker
import math
import pandas as pd
import pickle
import os, sys, time
import threading

protopath = "MobileNetSSD_deploy.prototxt"
faceprotopath = "deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
facemodelpath = "res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
facedetector = cv2.dnn.readNetFromCaffe(faceprotopath, facemodelpath)
embedder = cv2.dnn.readNetFromTorch("nn4.small2.v1.t7")
recognizer = pickle.loads(open("recognizer.pickle", "rb").read())
le = pickle.loads(open("le.pickle", "rb").read())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)

def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

dtime = dict()
dtime_a1 = dict()
dtime_a2 = dict()
dwell_time = dict()
dwell_time_area1 = dict()
dwell_time_area2 = dict()
faces = dict()
cust_name = dict()

test = cv2.VideoCapture('Testwface.mp4')
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("output10.avi", fourcc, 30, 
    (480, 270), True)
writertest = cv2.VideoWriter("test10.avi", fourcc, 30, 
    (480, 270), True)

path = './face/'
files = os.listdir(path)

def face_video_recognition():
    unknown_count = 1
    before_unknown = 0
    before = 'known'
    testing = 'known'
    name_count = dict()
    while True:
        retro,frames = test.read()
        if retro:
            frames = cv2.resize(frames,(480,270), cv2.INTER_CUBIC)
            (H,W) = frames.shape[:2]
            imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frames, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            facedetector.setInput(imageBlob)
            detections = facedetector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections
                if confidence > 0.6:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")
                    # extract the face ROI
                    face = frames[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]
                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue   
                    
                    # construct a blob for the face ROI, then pass the blob
                    # through our face embedding model to obtain the 128-d
                    # quantification of the face
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # perform classification to recognize the face
                    preds = recognizer.predict_proba(vec)[0]
                    j = np.argmax(preds)
                    proba = preds[j]
                    name = le.classes_[j]

                    # draw the bounding box of the face along with the
                    # associated probability
                    text = "{}: {:.2f}%".format(name, proba * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frames, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    cv2.putText(frames, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    
                    if name == 'Unknown':
                        if os.path.isfile('./face/Unknown_(%d).jpg' % unknown_count) is False:
                            if before_unknown != unknown_count:
                                cv2.imwrite("./face/Unknown_(%d).jpg" % unknown_count, frames)
                                before_unknown = unknown_count
                            else:
                                if before != 'Unknown':
                                    unknown_count+= 1
                                    before = name
                            print (before, testing)
                        
                    else:   
                        if before != name:
                            if testing != name:
                                name_count[name] = 0
                                testing = name
                                print (before, testing)
                                
                            else:
                                if os.path.isfile('./face/%s_(%d).jpg' % (name,name_count[name])) is False:
                                    cv2.imwrite("./face/%s_(%d).jpg" % (name, name_count[name]), frames)
                                    before = name
                                    print(before, testing)
                                    name_count[name] += 1
                                

                        else: continue

            writertest.write(frames)
            cv2.imshow("Frame", frames)

            key = cv2.waitKey(1) & 0xFF

                    # if the `` key was pressed, break from the loop
            if key == ord('z'):
                    break
        else: break
    writertest.release()
    test.release()


def cust_tracking():
    fps_start_time = datetime.datetime.now()
    total_frames = 0
    skip_frame = 1
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    object_id_area1_list = []
    object_id_curr_area1 = []
    object_id_area2_list = []
    object_id_curr_area2 = []
    area1_count = 0
    area2_count = 0
    valid_person_area1 = []
    valid_person_area2 = []
    exitpath_list = []
    cap = cv2.VideoCapture('testvideo3.mp4')

    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame,(480,270), cv2.INTER_CUBIC)
            (H,W) = frame.shape[:2]

            #declare VideoWrite outside loop
        
            if total_frames % skip_frame == 0:
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

                detector.setInput(blob)
                person_detections = detector.forward()
                rects = []
                for i in np.arange(0, person_detections.shape[2]):
                    confidence = person_detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(person_detections[0, 0, i, 1])

                        if CLASSES[idx] != "person":
                            continue

                        person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = person_box.astype("int")
                        rects.append(person_box)

                boundingboxes = np.array(rects)
                boundingboxes = boundingboxes.astype(int)
                rects = non_max_suppression_fast(boundingboxes, 0.3)

                objects = tracker.update(rects)
                for (objectId, bbox) in objects.items():
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    midx = int((x1+x2)/2)
                    midy = int((y1+y2)/2)

                    valid_person_area1.append((midx, midy, object_id_area1_list))
                    valid_person_area2.append((midx, midy, object_id_area2_list))
                    
                    if midy > 215 :
                        if objectId not in exitpath_list:
                            exitpath_list.append(objectId)

                    if objectId not in object_id_list:
                        object_id_list.append(objectId)
                        dtime[objectId] = datetime.datetime.now()
                        dwell_time[objectId] = 0
                    else:
                        curr_time = datetime.datetime.now()
                        old_time = dtime[objectId]
                        time_diff = curr_time - old_time
                        dtime[objectId] = datetime.datetime.now()
                        sec = time_diff.total_seconds()
                        dwell_time[objectId] += sec

                    for (midx, midy, object_id_area1_list)in valid_person_area1:
                        if 0 < midy < 215 and 37 < midx < 188:
                            if objectId not in object_id_area1_list:
                                area1_count += 1
                                object_id_area1_list.append (objectId)
                                object_id_curr_area1.append (objectId)
                                dtime_a1[objectId] = datetime.datetime.now()
                                dwell_time_area1[objectId] = 0
                        else:
                            if objectId in object_id_curr_area1:
                                object_id_curr_area1.remove(objectId)
                        valid_person_area1.remove((midx, midy, object_id_area1_list))

                    for (midx, midy, object_id_area2_list)in valid_person_area2:    
                        if 209 < midx < 443 and 100 < midy < 214:
                            if objectId not in object_id_area2_list:
                                area2_count += 1
                                object_id_area2_list.append (objectId)
                                object_id_curr_area2.append (objectId)
                                dtime_a2[objectId] = datetime.datetime.now()
                                dwell_time_area2[objectId] = 0
                        else:
                            if objectId in object_id_curr_area2:
                                object_id_curr_area2.remove (objectId)
                        valid_person_area2.remove((midx, midy, object_id_area2_list))    
                        
                    for objectId in dwell_time_area1:
                        if objectId in object_id_curr_area1:
                            curr_time_a1 = datetime.datetime.now()
                            old_time_a1 = dtime_a1[objectId]
                            time_diff_a1 = curr_time_a1 - old_time_a1
                            dtime_a1[objectId] = datetime.datetime.now()
                            sec_a1 = time_diff_a1.total_seconds()
                            dwell_time_area1[objectId] += sec_a1

                    for objectId in dwell_time_area2:
                        if objectId in object_id_curr_area2:
                            curr_time_a2 = datetime.datetime.now()
                            old_time_a2 = dtime_a2[objectId]
                            time_diff_a2 = curr_time_a2 - old_time_a2
                            dtime_a2[objectId] = datetime.datetime.now()
                            sec_a2 = time_diff_a2.total_seconds()
                            dwell_time_area2[objectId] += sec_a2
            
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = "{}|{}".format(objectId, int(dwell_time[objectId]))
                    cv2.circle(frame, (midx,midy),5,(0,0,255),2)
                    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                fps_end_time = datetime.datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)

                fps_text = "FPS: {:.2f}".format(fps)

                cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                lpc_count = len(objects)
                opc_count = len(object_id_list)

                lpc_txt = "LPC: {}".format(lpc_count)
                opc_txt = "OPC: {}".format(opc_count)
                area1_txt = "area1: {}".format(area1_count)
                area2_txt = "area2: {}".format(area2_count)

                cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, opc_txt, (5, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, area1_txt, (5, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.putText(frame, area2_txt, (5, 140), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
                cv2.rectangle(frame, (37, 0), (188, 214), (0, 255, 0), 2)
                cv2.rectangle(frame, (209, 100), (443, 214), (0, 255, 0), 2)
            
                writer.write(frame)

                cv2.imshow("Application", frame)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break

                total_frames = total_frames + 1

            else:
                break
        
        else:
            break

    for objectId in dwell_time:
        dwell_time[objectId]= round(dwell_time[objectId], 2)
    print (dwell_time)

    for objectId in dwell_time_area1:
        dwell_time_area1[objectId]= round(dwell_time_area1[objectId], 2)
    print (dwell_time_area1)

    for objectId in dwell_time_area2:
        dwell_time_area2[objectId]= round(dwell_time_area2[objectId], 2)
    print (dwell_time_area2)
    
    name_list = os.listdir(path)
    full_list = [os.path.join(path,i) for i in name_list]
    time_sorted_list = sorted(full_list, key=os.path.getmtime)
    print (time_sorted_list)

    for objectId, file in zip(exitpath_list, time_sorted_list):
        base = os.path.basename(file)
        cust_name[objectId] = os.path.splitext(base)[0]
    print (cust_name)
    
    fillexcel(dwell_time, dwell_time_area1, dwell_time_area2, cust_name)
    writer.release()
    cap.release()

def fillexcel(dwell_time, dwell_time_area1, dwell_time_area2, cust_name):
    overall_df = pd.DataFrame.from_dict(dwell_time, orient='index')
    area1_df = pd.DataFrame.from_dict(dwell_time_area1, orient='index')
    area2_df = pd.DataFrame.from_dict(dwell_time_area2, orient='index')
    customer_name_df = pd.DataFrame.from_dict(cust_name, orient='index')

    with pd.ExcelWriter('data10.xlsx') as writers:
        overall_df.to_excel(writers, sheet_name='Overall')
        area1_df.to_excel(writers, sheet_name='Area 1')
        area2_df.to_excel(writers, sheet_name='Area 2')
        customer_name_df.to_excel(writers, sheet_name='Customer Name')

if __name__ == '__main__':
    p1 = threading.Thread(target=face_video_recognition)
    p1.start()
    p = threading.Thread(target=cust_tracking)
    p.start()

cv2.destroyAllWindows()
