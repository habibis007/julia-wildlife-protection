from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import datetime
import cv2
import numpy as np
import os, model
import pandas as pd
import flask
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
import io
import imutils
from werkzeug.utils import secure_filename
import keras
from keras.models import load_model
# from datetime import datetime
import requests
from tensorflow.python.keras.backend import set_session
import ffmpeg
from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection

execution_path = os.getcwd() 
save_path = os.getcwd() + "/static/outputs"

net = None
animalGraph = None
fireNet = None
graph = None
sess = None
# initialize flask application and the keras model
app = flask.Flask(__name__)
UPLOAD_FOLDER='static'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['avi','mpeg','mp4','jpeg','jpg','tiff'])
labelPath = os.path.join("../models/yolo-coco/coco.names")
destination = "/home/habeeb-mikro/Desktop/Projects/JuliaAI/Wildlife-protection/web/static/outputs/"
LABELS = open(labelPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0,255, size=(len(LABELS), 3), dtype="uint8")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def loadObjectModel():
    global net
    yoloweight = "/home/habeeb-mikro/Desktop/Projects/JuliaAI/Wildlife-protection/models/yolo-coco/yolov3.weights"
    yolocfg = "/home/habeeb-mikro/Desktop/Projects/JuliaAI/Wildlife-protection/models/yolo-coco/yolov3.cfg"
    weigthsPath = os.path.sep.join("models/yolo-coco/yolov3.weights")
    configPath = os.path.sep.join("models/yolo-coco/yolov3.cfg")
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(yolocfg, yoloweight)

def loadAnimalModel():
    global animalGraph
    animalGraph = model.load_graph('static/output_graph.pb')

# def loadFireModel():
#     global fireNet
#     fireNet = CustomVideoObjectDetection()
#     fireNet.setModelTypeAsYOLOv3()
#     fireNet.setModelPath(detection_model_path=os.path.join(execution_path, "static/fire/detection_model-ex-33--loss-4.97.h5"))
#     fireNet.setJsonPath(configuration_json=os.path.join(execution_path, "static/fire/detection_config.json"))
#     fireNet.loadModel()


def processVideoCompress(filepath):
    dest = None
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #initialize the video stream, pointer to output file, and
    # frame dimensions
    vs = cv2.VideoCapture(filepath)
    writer = None
    (W, H) = (None, None)
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total frames
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, end of stream reached
        if not grabbed:
            break
        # if the frame dimensions are empty, grab frame dimension
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # construct a blob from the input frame and then perform a
        # forward pass of the YOLO object detector, returning bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize the lists of detected bounding boxes,
        # confidences,
        # and class IDs
        boxes = []
        confidences = []
        classIDs = []

        #loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the
                # detected probability is greater than the 
                # minimum confidence
                if confidence > 0.6:
                    box = detection[0:4] * np.array([W,H,W,H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update the list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x,y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima supression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes kept
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # check if the video writer is None
        if writer is None:
            # initialize video writer
            output_file = datetime.datetime.now().isoformat() + ".avi"
            output = os.path.join(destination, output_file)
            dest = output_file
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # writer = cv2.VideoWriter(output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            writer = (
                ffmpeg
                    .input('pipe:',format='rawvideo', pix_fmt='rgb24',s='{}x{}'.format(frame.shape[1], frame.shape[0]))
                    .output(output,pix_fmt='yuv420p',r=30, crf=37)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)   
            )
            
            # if not writer.isOpened():
            #     raise IOError("Unable to open video stream")
            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
        #write the output frame to disk
        writer.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8).tobytes())

        
        # writer.write(frame)
    # release the file pointers

    
    print("[INFO] cleaning up after task...")
    # print(writer)
    # writer.release()
    vs.release()
    response = True
    return response, dest

def processVideoUncompress(filepath):
    dest = None
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #initialize the video stream, pointer to output file, and
    # frame dimensions
    vs = cv2.VideoCapture(filepath)
    writer = None
    (W, H) = (None, None)
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))

    # an error occurred while trying to determine the total frames
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        # if the frame was not grabbed, end of stream reached
        if not grabbed:
            break
        # if the frame dimensions are empty, grab frame dimension
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # construct a blob from the input frame and then perform a
        # forward pass of the YOLO object detector, returning bounding boxes
        # and associated probabilities
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        # initialize the lists of detected bounding boxes,
        # confidences,
        # and class IDs
        boxes = []
        confidences = []
        classIDs = []

        #loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the
                # detected probability is greater than the 
                # minimum confidence
                if confidence > 0.6:
                    box = detection[0:4] * np.array([W,H,W,H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update the list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x,y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima supression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.3)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes kept
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x,y) = (boxes[i][0], boxes[i][1])
                (w,h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # check if the video writer is None
        if writer is None:
            # initialize video writer
            output_file = datetime.datetime.now().isoformat() + ".avi"
            output = os.path.join(destination, output_file)
            dest = output_file
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            # writer = cv2.VideoWriter(output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)
            writer = (
                ffmpeg
                    .input('pipe:',format='rawvideo', pix_fmt='rgb24',s='{}x{}'.format(frame.shape[1], frame.shape[0]))
                    .output(output,pix_fmt='yuv420p', vcodec='rawvideo', r=30, crf=37)
                    .overwrite_output()
                    .run_async(pipe_stdin=True)   
            )
            
            # if not writer.isOpened():
            #     raise IOError("Unable to open video stream")
            # some information on processing single frame
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
        #write the output frame to disk
        writer.stdin.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8).tobytes())

        
        # writer.write(frame)
    # release the file pointers

    
    print("[INFO] cleaning up after task...")
    # print(writer)
    # writer.release()
    vs.release()
    response = True
    return response, dest

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

    
# GET method displays page
# POST upload video file
@app.route('/upload/compress', methods=['GET', 'POST'])
def predictCompress():
    if request.method == 'POST':
        # check if request has file
        if 'file' not in request.files:
            response = jsonify({'message': 'No file provided'})
            response.status_code = 400
            return response

        # Get the file from post request
        f = request.files['file']
        if f.filename == '':
            response = jsonify({'message': 'No file selected for upload'})
            response.status_code = 400
            return response
        
        if f and allowed_file(f.filename):
            # Save file to ./uploads folder
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            #process the video file and generate object detection output
            state, dest = processVideoCompress(file_path)
            if state == True:
                response = jsonify({'message': 'Processing successful', 'destination': dest})
                response.status_code = 200
                return response
            else:
                response = jsonify({'message': 'Something went wrong'})
                response.status_code = 400
                return response
        else:
            response = jsonify({'message': 'Allowed file types are avi, mpeg, and mp4'})
            response.status_code = 400
            return response


def detect_from_video(file):
    fireNet = CustomVideoObjectDetection()
    fireNet.setModelTypeAsYOLOv3()
    fireNet.setModelPath(detection_model_path=os.path.join(execution_path, "static/fire/detection_model-ex-33--loss-4.97.h5"))
    fireNet.setJsonPath(configuration_json=os.path.join(execution_path, "static/fire/detection_config.json"))
    fireNet.loadModel()
    detected_video_path = fireNet.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, file), frames_per_second=30, output_file_path=os.path.join(save_path, "video1-detected"), minimum_percentage_probability=40, log_progress=True )
    return detected_video_path


# POST upload video file
@app.route('/upload/uncompress', methods=['GET', 'POST'])
def predictUncompress():
    if request.method == 'POST':
        # check if request has file
        if 'file' not in request.files:
            response = jsonify({'message': 'No file provided'})
            response.status_code = 400
            return response

        # Get the file from post request
        f = request.files['file']
        if f.filename == '':
            response = jsonify({'message': 'No file selected for upload'})
            response.status_code = 400
            return response
        
        if f and allowed_file(f.filename):
            # Save file to ./uploads folder
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            #process the video file and generate object detection output
            state, dest = processVideoUncompress(file_path)
            if state == True:
                response = jsonify({'message': 'Processing successful', 'destination': dest})
                response.status_code = 200
                return response
            else:
                response = jsonify({'message': 'Something went wrong'})
                response.status_code = 400
                return response
        else:
            response = jsonify({'message': 'Allowed file types are avi, mpeg, and mp4'})
            response.status_code = 400
            return response

#POST upload video - fire
@app.route('/fire', methods=['GET','POST'])
def predictFire():
    if request.method == 'POST':
        # check if request has file
        if 'file' not in request.files:
            response = jsonify({'message': 'No file provided'})
            response.status_code = 400
            return response

        # Get the file from post request
        f = request.files['file']
        if f.filename == '':
            response = jsonify({'message': 'No file selected for upload'})
            response.status_code = 400
            return response
        
        if f and allowed_file(f.filename):
            # Save file to ./uploads folder
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            #process the video file and generate object detection output
            dest = detect_from_video(file_path)
            head, tail = os.path.split(dest)
            print(tail)
            if(tail):
                response = jsonify({'message': 'successful', 'destination': tail})
                response.status_code = 200
                return response
        else:
            response = jsonify({'message': 'Something went wrong'})
            response.status_code = 400
            return response



#POST Animal image
@app.route("/animal",methods=['GET','POST'])
def results():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        # image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))

        label_file = 'static/output_labels.txt'
        input_layer = "Placeholder"
        output_layer = "final_result"

        t = model.read_tensor_from_image_file(file_path,299,299,0,255)
        input_name = input_layer
        output_name = output_layer
        input_operation = animalGraph.get_operation_by_name(input_name)
        output_operation = animalGraph.get_operation_by_name(output_name)

        with tf.compat.v1.Session(graph=animalGraph) as sess:
            results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
            }) 
        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]
        labels = model.load_labels(label_file)
        response = {}
        for i in top_k:
            response.update({labels[i]:str(results[i] * 100)})

        return jsonify(response)

    return "Error"




# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Object detection model..."
    "please wait until server has fully started"))
    loadObjectModel()
    print(("* Loading Animal classification model..."
    "please wait until server has fully started"))
    loadAnimalModel()
    print(("* Loading Fire detction model..."
    "please wait until server has fully started"))
    # loadFireModel()
    print("Application started successfully!")
    app.run(port=5000, debug=True)



