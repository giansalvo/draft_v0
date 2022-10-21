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
import os
from django.http import HttpResponse # gians TODELETE DEBUG
from django.shortcuts import render, redirect
import logging 
from uploadimg.forms import ImageForm
from upload.settings import LOGGING
import datetime

# gians: libraries for AI
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_addons as tfa
import cv2

IMAGE_TEMP = "img.jpg"
MODEL_PATH = "model_unet_us_w46.h5"


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
            img = cv2.imread(IMAGE_TEMP)

            print("Loading network model from: " + MODEL_PATH)
            model = tf.keras.models.load_model(MODEL_PATH)

            overlay = park_detection(model, img)

            fname = os.path.join(img_output_path, img_output_fname)
            print("Saving trimap output segmented image to file: " + fname)
            cv2.imwrite(fname, overlay,  [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            
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


def image_fusion(background, foreground, sharp=False, alfa=0.5):
    img1 = background
    img2 = foreground

    rows,cols,channels = img2.shape
    roi = img1[0:rows, 0:cols ]

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_hidden_bg = cv2.bitwise_and(img1, img1, mask = mask)
    #cv2.imshow("img1_hidden_bg", img1_hidden_bg)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    #cv2.imshow("img1_bg", img1_bg)

    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    #cv2.imshow("img2_fg", img2_fg)

    out_img = cv2.add(img1_bg,img2_fg)
    #cv2.imshow("out_img", out_img)
    img1[0:rows, 0:cols ] = out_img

    # cv2.imshow("Result Sharp", img1)

    ALFA = 0.5
    img1_hidden_merge = cv2.addWeighted(img1_hidden_bg, ALFA, img2_fg, 1-ALFA, 0.0)
    img_opaque = cv2.add(img1_bg, img1_hidden_merge)

    if sharp:
        return img1
    else:
        return img_opaque


def park_detection(model, img):
    global logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('gians')

    w_orig = img.shape[1]
    h_orig = img.shape[0]

    # img0, _, _ = read_image("I000.jpg") 
    img0 = tf.image.resize(img, [256, 256]) # TODO HARDCODED VALUEs
    img_tensor = tf.cast(img0, tf.float32) / 255.0    # normalize
    img_tensor = np.expand_dims(img_tensor, axis=0)

    predictions = model.predict(img_tensor)
    pred = create_mask(predictions)[0]
    pred = pred + 1 # de-normalization

    # plot_samples_matplotlib([img0, pred], ["sample", "prediction"] )

    # convert to OpenCV image format
    i0 = img0.numpy()
    i1 = pred.numpy()
    i1 = np.squeeze(i1)
    i1 = np.float32(i1)

    # overlay = get_overlay(i0, i1)
    
    FOREGROUND = 1
    OFFSET = 255 - FOREGROUND
    RED = [0,0, 255] # BGR
    WHITE = [255,255,255]

    #####
    # fusion of sample image and foreground area from predicted image
    #####
    img_pred = i1
    img_pred += OFFSET
    img_pred=cv2.cvtColor(img_pred, cv2.COLOR_GRAY2BGR)
    img_pred[np.all(img_pred == WHITE, axis=-1)] = RED
    i1 = img_pred

    i0 = i0.astype(np.uint8)
    i1 = i1.astype(np.uint8)
   
    overlay = image_fusion(i0, i1)

    overlay = cv2.resize(overlay, (w_orig, h_orig))

    return overlay


def temp_orig():
    # Read the graph.
    with tf.io.gfile.GFile('frozen_graph_unet.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        w_orig = img.shape[1]
        h_orig = img.shape[0]
        img_input = cv2.resize(img, (256, 256))  # TODO HARDCODED VALUE IN THE GRAPH 256x256
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
        pred = (pred + 1) * 100 # TODO HARDCODED JUST TO MAKE VISIBLE WHEN DISPLAYING ON WEB

        img_input = tf.squeeze(img_input)
        print(img_input.shape)
        print(pred.shape)
        overlay = tfa.image.blend(img_input, pred, 0.5)
        overlay = cv2.resize(img, (w_orig, h_orig))


        #cv.imwrite(img_output_path + img_output_fname, pred)
        fname = img_output_path + img_output_fname
        print("Saving trimap output segmented image to file: " + fname)
        img1 = tf.cast(overlay, tf.uint8)
        img1 = tf.image.encode_jpeg(img1)
        fwrite = tf.io.write_file(tf.constant(fname), img1)
        result = sess.run(fwrite)

    return
