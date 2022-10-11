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


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [img_size, img_size, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [img_size, img_size, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [img_size, img_size, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [img_size, img_size]
    # but matplotlib needs [img_size, img_size, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask

def park_detection(img, img_output_path, img_output_fname):
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')
    template_name = 'main/index.html'

    tf.disable_v2_behavior()

    # Read the graph.
    with tf.io.gfile.GFile('frozen_graph_unet.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        img_input = cv.resize(img, (256, 256))  # TODO HARDCODED VALUE IN THE GRAPH 256x256
        # img_input = img_input[:, :, [2, 1, 0]]  # BGR2RGB
        img_input = np.expand_dims(img_input, axis=0)

        # logger.info("park_detection: print operations")
        # for item in sess.graph.get_operations():
        #     print(str(item.name))

        # Run the model
        tensor_output = sess.graph.get_tensor_by_name('Identity:0')
        tensor_input = sess.graph.get_tensor_by_name('x:0')
        # inference = sess.run(tensor_output, {tensor_input:img_input})
        inference = sess.run(tensor_output, 
                            feed_dict={tensor_input: img_input})

        predictions = create_mask(inference)
        pred = predictions[0]
        pred *= 100 # TODO HARDCODED JUST TO MAKE VISIBLE WHEN DISPLAYING ON WEB

        logger.info("park_detection: predicted mask")
        print(pred)
    
        #cv.imwrite(img_output_path + img_output_fname, pred)
        fname = img_output_path + img_output_fname
        print("Saving trimap output segmented image to file: " + fname)
        img1 = tf.cast(pred, tf.uint8)
        img1 = tf.image.encode_jpeg(img1)
        fwrite = tf.io.write_file(tf.constant(fname), img1)
        result = sess.run(fwrite)

    return
