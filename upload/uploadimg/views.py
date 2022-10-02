"""
    Simple project for CNN demonstration

    Copyright (c) 2022 Giansalvo Gusinu

    Permission is hereby granted, free of charge, to any person obtaining a 
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""
from django.http import HttpResponse # gians TODELETE DEBUG
from django.shortcuts import render, redirect
import logging 
from uploadimg.forms import ImageForm
from upload.settings import LOGGING
import datetime

# gians: libraries for AI
import numpy as np
import tensorflow.compat.v1 as tf
import cv2 as cv

IMAGE_TEMP = "img.jpg"

def handl_uploaded_file(f, fname=IMAGE_TEMP):
	with open(fname, 'wb+') as destination:
		for chunk in f.chunks():
			destination.write(chunk)

def index(request):
    """Process images uploaded by users"""

    # create logger
    logger = logging.getLogger('gians')
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s:%(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    logger.info("Starting")
    
    # TODO use name of input file and use dynamic
    img_output_path = "./uploadimg/static/"
    
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
                        
            # Get the current instance object to display in the template
            img_obj = form.instance
            img_output_fname = img_obj.title + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jpg"

            handl_uploaded_file(request.FILES['image'])
            img = cv.imread(IMAGE_TEMP)

            park_detection(img, img_output_path, img_output_fname)
            
            try:
                img_output = open(img_output_path + img_output_fname, "rb")
                logger.info("File loaded.")
                logger.info(img_output)
            except IOError:
                logger.info("IOError")
                return HttpResponse("File not found")
                #TODO improve error page 

            return render(request, 'index.html', {'form': form, 
                        'img_obj': img_obj, 
                        'img_output': "static/" + img_output_fname}) #TODO HARDCODED
    else:
        form = ImageForm()
    return render(request, 'index.html', {'form': form})


def park_detection(img, img_output_path, img_output_fname):
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')
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

    return
