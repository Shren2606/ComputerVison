
from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys
import time
import os

cap = cv2.VideoCapture(0)

# Dinh nghia class
class_name = ['A','green','red','yellow']

import serial

connected = False

ser =serial.Serial(port='/dev/ttyUSB0',
					baudrate = 9600,
					timeout = 5)
					
while not connected:
	serin =ser.read()
	connected = True

def get_model():
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    # Dong bang cac layer
    for layer in model_vgg16_conv.layers:
        layer.trainable = False

    # Tao model
    input = Input(shape=(128, 128, 3), name='image_input')
    output_vgg16_conv = model_vgg16_conv(input)

    # Them cac layer FC va Dropout
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(2048, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='softmax', name='predictions')(x)

    # Compile
    my_model = Model(inputs=input, outputs=x)
    my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model

# Load weights model da train
my_model = get_model()
my_model.load_weights("vggmodel.h5")


while(True):

	s= ser.readline()
	data = s.decode()
	data = data.rstrip()
	ret, image_org = cap.read()
	index = 0
	pp_arr = np.array([],dtype = int)
	cn = ""
	ser.write(str.encode('ok'))
	ser.flush()
	print("ok")

	if data == "oncam" :
		while index < 4 :
			ser.write(str.encode('ok'))
			ser.flush()
			print(data)
			ret, image_org = cap.read()
			image_org = cv2.resize(image_org, dsize=None,fx=1,fy=1)
			image_org = image_org[50:, :550]
			cv2.imwrite('data/stream/'+str(index)+".png",image_org)
			cv2.imwrite('data/Astream/'+str(index)+".png",image_org)
			index +=1
			print('while ' +str(index))

		image = cv2.imread('data/stream/' +'3'+'.png')
		image = cv2.resize(image, dsize=(128, 128))
		image = np.expand_dims(image, axis=0)
				
		predict = my_model.predict(image)
		cn =  class_name[np.argmax(predict[0])]
		predict_per = np.max(predict)
		#print('name' +str(a)+'.png')
			
				
		if (np.max(predict)>=0.8) and (np.argmax(predict[0])!=0):
			for b in range(0,3):
				os.remove('data/stream/' +str(b)+'.png')
			#break
		ser.readline()
		if cn=="green":
			print("green ok")
			ser.write(str.encode('xanh'))
			ser.flush()
			time.sleep(1)
		if cn =="red":
			print("red'ok")
			ser.write(str.encode('do'))
			ser.flush()
			time.sleep(1)
		if cn =="yellow":
			ser.write(str.encode('vang'))
			ser.flush()
			print("yellow ok")
			time.sleep(1)
		if cn=="A":
			print("A ok")

    # Predict
         
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
