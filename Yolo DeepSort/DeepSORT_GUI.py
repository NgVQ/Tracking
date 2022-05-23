
import tkinter
from tools import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import preprocessing, nn_matching
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import ConfigProto
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from core.yolov4 import filter_boxes
import core.utils as utils
from absl.flags import FLAGS
from absl import app, flags, logging
import tensorflow as tf
import time
from tkinter import *
from tkinter import filedialog
from tkinter import font
from PIL import Image, ImageTk
import cv2
from cv2 import VideoCapture
import imutils
import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# deep sort imports


max_cosine_distance = 0.4
nn_budget = None
nms_max_overlap = 1.0
saved_model_loaded = tf.saved_model.load(
    './yolov4', tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)
# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config()
input_size = 416
frame_num = 0
out = None

def track(frame):
    global frame_num,cap,out
    # get video ready to save locally
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./outputs/test.avi', codec, fps, (width, height))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    frame_num += 1
    print('Frame : ', frame_num)
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )
    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]
    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)
    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]

    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    # allowed_classes = list(class_names.values())

    # custom allowed classes (uncomment line below to customize tracker for only people)
    allowed_classes = ['person']
    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
    cv2.putText(frame, "Objects: {}".format(count), (5, 35),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    print("Objects being tracked: {}".format(count))
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)

    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox,
                  score, class_name, feature in zip(bboxes, scores, names, features)]

    # initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(
        boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()

    # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(
            bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
            len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id),
                    (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)
        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
            str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

    # calculate frames per second of running detections
    fps = 1.0 / (time.time() - start_time)
    print("FPS: %.2f" % fps)
    result = np.asarray(frame)
    out.write(result)
    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
    # draw bbox on screen
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
        cv2.rectangle(frame, (int(bbox[0]), int(
            bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(
            len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(frame, class_name + "-" + str(track.track_id),
                    (int(bbox[0]), int(bbox[1]-10)), 0, 0.75, (255, 255, 255), 2)
        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(
            str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
    return frame


def visualizer():
    global cap
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=900, height=600)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = track(frame)
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(5, visualizer)
    else:
        lblVideo.image = ""
        lblInfoVideoPath.configure(text="")
        rad1.configure(state="active")
        rad2.configure(state="active")
        selected.set(0)
        btnCloseTrack.configure(state="disabled")
        cap.release()


def btnClose():
    lblVideo.image = ""
    lblInfoVideoPath.configure(text="")
    rad1.configure(state="active")
    rad2.configure(state="active")
    selected.set(0)
    cap.release()


def openVideo():
    global cap
    if selected.get() == 1:
        filepath = filedialog.askopenfilename(
            filetypes=[("all video format", ".mp4"), ("all video format", ".avi")])
        if len(filepath) > 0:
            btnCloseTrack.config(state="active")
            rad1.config(state="disabled")
            rad2.config(state="disabled")
            pathInputVideo = "..." + filepath[-20:]
            lblInfoVideoPath.config(text=pathInputVideo)
            cap = VideoCapture(filepath)
            visualizer()
    if selected.get() == 2:
        btnCloseTrack.config(state="active")
        rad1.config(state="disabled")
        rad2.config(state="disabled")
        lblInfoVideoPath.config(text="CAMERA")
        cap = VideoCapture(0, cv2.CAP_DSHOW)
        visualizer()


cap = None
root = Tk()
window_width = 1280
window_height = 800

# get the screen dimension
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# find the center point
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - window_height / 2)

# set the position of the window to the center of the screen
root.title("Mulitply Object Tracking")
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
lblInfo1 = Label(root, text="YOLOv4 DeepSORT", font="bold")
lblInfo1.grid(column=3, row=0)

selected = IntVar()
rad1 = Radiobutton(root, text="Select video", width=20,
                   value=1, variable=selected, command=openVideo)
rad2 = Radiobutton(root, text="Open Camera", width=20,
                   value=2, variable=selected, command=openVideo)
rad1.grid(column=1, row=1, sticky= tkinter.W , pady= 10)
rad2.grid(column=1, row=2,sticky= tkinter.W, padx= 5, pady= 10)
lblInfoVideoPath = Label(root, text="", width=25)
lblInfoVideoPath.grid(column=3, row=1)
lblVideo = Label(root)
root.columnconfigure(3,weight= 12)
lblVideo.grid(column=3, row=2 , rowspan= 9)
btnCloseTrack = Button(root, text=" Close Traking",
                       state="disabled", command=btnClose)
btnCloseTrack.grid(column=1, row=3)
root.mainloop()
