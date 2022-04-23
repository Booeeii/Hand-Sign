# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 13:41:22 2022

@author: Windows 11
"""
#WHere I will keep user uploads
UPLOAD_FOLDER = 'static/uploads'
#Allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import io
import json

import torch.optim as optim


import matplotlib.pyplot as plt


#website library
from flask import render_template
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

#Load math library
import numpy as np

#Load machine learning libraries
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import load_model
#from tensorflow.compat.v1.keras.backend import set_session
import tensorflow as tf
import sklearn.neighbors as sn

from flask import Flask, jsonify, request

import pandas as pd

import csv

import cv2 as cv
#from keras.applications.vgg16 import preprocessing_input
#from keras_applications.vgg16 import decode_predictions
#Create the website object
app = Flask(__name__)
pathpic = "white.png"





"""
#load model from file
def load_model_from_file():
    #Set up the machine learning session
    mySession = tf.Session()
    set_session(mySession)
    myModel = load_model('อันนี้ไว้ใส่ชื่อ Model ที่เก็บอยู่ใน folder FDS HAND')
    myGraph = tf.get_default_graph()
    return (mySession,myModel,myGraph) 
"""


#Try to allow only images
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/do.html')
def updo():
    return render_template('do.html')

@app.route('/uploads/credit.html')
def upcredit():
    return render_template('credit.html')

@app.route('/uploads/index.html')
def upindex():
    return render_template('index.html', path=pathpic)



@app.route('/do.html')
def do():
    return render_template('do.html')

@app.route('/credit.html')
def credit():
    return render_template('credit.html')

@app.route('/index.html')
def index():
    return render_template('index.html', path=pathpic)

@app.route('/')
def first():
    return render_template('index.html', path=pathpic)


"""
@app.route('/', methods=['POST'])
def predict():
    image_file = request.files["file"]
    image_path = UPLOAD_FOLDER+"/"+image_file.filename
    image_file.save(image_path)
    
    image = load_img(image_path, target_size=(255, 255))
    image = img_to_array(image)
    results = model(image)
    #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    #image = preprocessing_input(image)
    #yhat = model.predict(image)
    #label = decode_predictions(yhat)
    #label = label[0][0]
    
    #classification = "%s (%.2f%%)" %(label[1], label[2]*100)
    
    return render_template('index.html', results = results)

"""
#Define the view for the top level page
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    #Initial webpage load
    if request.method == 'GET' :
        return render_template('index.html')
    else: # if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser may also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        # If it doesn't look like an image file
        if not allowed_file(file.filename):
            flash('I only accept files of type'+str(ALLOWED_EXTENSIONS))
            return redirect(request.url)
        #When the user uploads a file with good parameters
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))



@app.route('/uploads/<filename>')

def uploaded_file(filename):
    
    def get_data(filename):
      # You will need to write code that will read the file passed
      # into this function. The first line contains the column headers
      # so you should ignore it
      # Each successive line contians 785 comma separated values between 0 and 255
      # The first value is the label
      # The rest are the pixel values for that picture
      # The function will return 2 np.array types. One with all the labels
      # One with all the images
      #
      # Tips: 
      # If you read a full line (as 'row') then row[0] has the label
      # and row[1:785] has the 784 pixel values
      # Take a look at np.array_split to turn the 784 pixels into 28x28
      # You are reading in strings, but need the values to be floats
      # Check out np.array().astype for a conversion
        with open(filename) as training_file:
          csv_reader = csv.reader(training_file, delimiter=',')
          first_line = True
          temp_images = []
          temp_labels = []
          for row in csv_reader:
            if first_line:
              first_line = False
            else:
              temp_labels.append(row[0])
              images_data = row[1:785]
              images_data_as_array = np.array_split(images_data, 28)
              temp_images.append(images_data_as_array)
              
          images = np.array(temp_images).astype('float')
          labels = np.array(temp_labels).astype('str')
          
        return images, labels
    
    #training_images, training_labels = get_data('/content/drive/MyDrive/Sign Language MNIST/Data/Train/sign_mnist_train_alp.csv')
    testing_images, testing_labels = get_data('E:/FDS HAND/input/sign_mnist_test/sign_mnist_test_alp.csv')
    
    enc = dict()
    dec = dict()
    labeluni = np.unique(testing_labels)
    for idx, val in enumerate(labeluni):
        enc[val] = idx
        dec[idx] = val
        
    target_test = np.zeros(testing_labels.shape[0])
    for idx, val in enumerate(testing_labels):
        target_test[idx] = enc[val]


    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(24, activation='softmax')
    ])

    model.load_weights("E:/FDS HAND/model_best.h5")
    
    
    # Compile Model. 
    model.compile(optimizer = 'adam',
                  loss ='sparse_categorical_crossentropy',
                  metrics = ['acc']) #

    
    
    testing_images = np.expand_dims(testing_images,axis=3)
    
    model.evaluate(testing_images, target_test) 
    
    path = UPLOAD_FOLDER+"/"+filename
    pathpic = path[7:]
    img = cv.imread(path, 0)
    img= cv.resize(img, (28,28))
    
    img = np.expand_dims(img ,axis=2)
    img = np.expand_dims(img ,axis=0)
    
    result = model.predict(img)
    ans = dec[np.argmax(result)]#แปลงให้กลับมาเป็นตัวหนังสือ
    
    return render_template('index.html',results=ans, path=pathpic)
 



def main():
    #(mySession, myModel, myGraph) = load_model_from_file()
    
    app.config['SECRET_KEY'] = 'super secret key'
    
    """
    app.config['SESSION'] = mySession
    app.config['Model'] = myModel
    app.config['Graph'] = myGraph
    """
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16*1024*1024 #16MB upload limit
    app.run()

#Create a running list of results
results = []

#Lauch everything
main()