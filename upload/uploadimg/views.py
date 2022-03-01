from http.client import HTTPResponse # gians TODELETE DEBUG
from django.http import HttpResponse # gians TODELETE DEBUG
from django.shortcuts import render, redirect
import logging # gians LOG TODO not working properly

from uploadimg.forms import ImageForm

# gians: libraries for AI
import numpy as np
import tensorflow.compat.v1 as tf
import cv2 as cv
#from tkinter import Image # gians TODELETE???

def index(request):
    """Process images uploaded by users"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')
    
    # TODO use name of input file and use dynamic
    img_output_path = "./uploadimg/static/"
    
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
                        
            # Get the current instance object to display in the template
            img_obj = form.instance
            
            img_output_fname = img_obj.title + "_new.jpg"

            park_detection(img_obj, img_output_path, img_output_fname)
            
            try:
                img_output = open(img_output_path + img_output_fname, "rb")
                logger.info("File loaded.")
                logger.info(img_output)
            except IOError:
                logger.info("IOError")
                return HttpResponse("File not found")
                #TODO improve error page 

            return render(request, 'index.html', {'form': form, 'img_obj': img_obj, 'img_output': "static/" + img_output_fname})
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


def park_detection(img_obj, img_output_path, img_output_fname):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')
    template_name = 'main/index.html'

    # Example for Tensorflow 2.0
    tf.disable_v2_behavior()

    logger.info("park_detection()") #TODO not working
    logger.info(img_output_fname) #TODO not working

    # Read the graph.
    with tf.io.gfile.GFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = cv.imread('example.jpg')
        #img = cv.imread(img_obj)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    # cv.imshow('TensorFlow MobileNet-SSD', img)
    # cv.waitKey()
    
    cv.imwrite(img_output_path + img_output_fname, img)
    cv.imwrite("newimage.jpg", img) #TODELETE HARDCODED DEBUG

    return

def index_orig_with_tf(request):
    template_name = 'main/index.html'
    # Example for Tensorflow 2.0

    tf.disable_v2_behavior()

    # Read the graph.
    with tf.io.gfile.GFile('frozen_inference_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = cv.imread('example.jpg')
        img_orig = cv.imread('example.jpg')
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                    feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

    # cv.imshow('TensorFlow MobileNet-SSD', img)
    # cv.waitKey()
    
    cv.imwrite("newimage.jpg", img)

    imencoded = cv.imencode(".jpg", img)[1]
    
    return render(request, 'index.html', {'img_obj': img})

    try:
        return HttpResponse("example.jpg", content_type="image/jpeg")
    except IOError:
        return HttpResponse("errore")
