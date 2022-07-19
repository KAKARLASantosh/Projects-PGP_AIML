# Import required libraries
from __future__ import division
import sys
from tkinter import *
from random import *
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import os
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline
from skimage.transform import resize
from skimage.io import imread
import tensorflow as tf
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import pprint
import time
from optparse import OptionParser
import pickle
from keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from keras_frcnn import config, data_generators
#from keras_frcnn import losses_cust
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils
from keras_frcnn.simple_parser import get_data
import csv
from keras import backend as K
from tensorflow.keras.metrics import categorical_crossentropy
import cv2
import numpy as np
import csv
from keras.utils import generic_utils
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models

global class_mapping
class_mapping = {0: 'Audi TTS Coupe 2012', 1: 'Acura TL Sedan 2012', 2: 'Dodge Dakota Club Cab 2007', 3: 'Hyundai Sonata Hybrid Sedan 2012', 4: 'Ford F-450 Super Duty Crew Cab 2012', 5: 'Geo Metro Convertible 1993', 6: 'Dodge Journey SUV 2012', 7: 'Dodge Charger Sedan 2012', 8: 'Mitsubishi Lancer Sedan 2012', 9: 'Chevrolet Traverse SUV 2012', 10: 'Buick Verano Sedan 2012', 11: 'Toyota Sequoia SUV 2012', 12: 'Hyundai Elantra Sedan 2007', 13: 'Dodge Caravan Minivan 1997', 14: 'Volvo C30 Hatchback 2012', 15: 'Plymouth Neon Coupe 1999', 16: 'Chevrolet Malibu Sedan 2007', 17: 'Volkswagen Beetle Hatchback 2012', 18: 'Chevrolet Corvette Ron Fellows Edition Z06 2007', 19: 'Chrysler 300 SRT-8 2010', 20: 'BMW M6 Convertible 2010', 21: 'GMC Yukon Hybrid SUV 2012', 22: 'Nissan Juke Hatchback 2012', 23: 'Volvo 240 Sedan 1993', 24: 'Suzuki SX4 Sedan 2012', 25: 'Dodge Ram Pickup 3500 Crew Cab 2010', 26: 'Spyker C8 Coupe 2009', 27: 'Land Rover Range Rover SUV 2012', 28: 'Hyundai Elantra Touring Hatchback 2012', 29: 'Chevrolet Cobalt SS 2010', 30: 'Hyundai Veracruz SUV 2012', 31: 'Ferrari 458 Italia Coupe 2012', 32: 'BMW Z4 Convertible 2012', 33: 'Dodge Charger SRT-8 2009', 34: 'Fisker Karma Sedan 2012', 35: 'Infiniti QX56 SUV 2011', 36: 'Audi A5 Coupe 2012', 37: 'Volkswagen Golf Hatchback 1991', 38: 'GMC Savana Van 2012', 39: 'Audi TT RS Coupe 2012', 40: 'Rolls-Royce Phantom Sedan 2012', 41: 'Porsche Panamera Sedan 2012', 42: 'Bentley Continental GT Coupe 2012', 43: 'Jeep Grand Cherokee SUV 2012', 44: 'Audi R8 Coupe 2012', 45: 'Cadillac Escalade EXT Crew Cab 2007', 46: 'Bentley Continental Flying Spur Sedan 2007', 47: 'Chevrolet Avalanche Crew Cab 2012', 48: 'Dodge Dakota Crew Cab 2010', 49: 'HUMMER H3T Crew Cab 2010', 50: 'Ford F-150 Regular Cab 2007', 51: 'Volkswagen Golf Hatchback 2012', 52: 'Ferrari FF Coupe 2012', 53: 'Toyota Camry Sedan 2012', 54: 'Aston Martin V8 Vantage Convertible 2012', 55: 'Audi 100 Sedan 1994', 56: 'Ford Ranger SuperCab 2011', 57: 'GMC Canyon Extended Cab 2012', 58: 'Acura TSX Sedan 2012', 59: 'BMW 3 Series Sedan 2012', 60: 'Honda Odyssey Minivan 2012', 61: 'Dodge Durango SUV 2012', 62: 'Toyota Corolla Sedan 2012', 63: 'Chevrolet Camaro Convertible 2012', 64: 'Ford Edge SUV 2012', 65: 'Bentley Continental GT Coupe 2007', 66: 'Audi 100 Wagon 1994', 67: 'Ford E-Series Wagon Van 2012', 68: 'Jeep Patriot SUV 2012', 69: 'Audi S6 Sedan 2011', 70: 'Mercedes-Benz S-Class Sedan 2012', 71: 'Hyundai Sonata Sedan 2012', 72: 'Rolls-Royce Phantom Drophead Coupe Convertible 2012', 73: 'Ford GT Coupe 2006', 74: 'Cadillac CTS-V Sedan 2012', 75: 'BMW X3 SUV 2012', 76: 'Chevrolet Express Van 2007', 77: 'Chevrolet Impala Sedan 2007', 78: 'Chevrolet Silverado 1500 Extended Cab 2012', 79: 'Mercedes-Benz C-Class Sedan 2012', 80: 'Hyundai Santa Fe SUV 2012', 81: 'Dodge Sprinter Cargo Van 2009', 82: 'GMC Acadia SUV 2012', 83: 'Hyundai Genesis Sedan 2012', 84: 'Dodge Caliber Wagon 2012', 85: 'Jeep Liberty SUV 2012', 86: 'Mercedes-Benz 300-Class Convertible 1993', 87: 'Ford Expedition EL SUV 2009', 88: 'BMW 1 Series Coupe 2012', 89: 'Jaguar XK XKR 2012', 90: 'Hyundai Accent Sedan 2012', 91: 'Isuzu Ascender SUV 2008', 92: 'Nissan 240SX Coupe 1998', 93: 'Scion xD Hatchback 2012', 94: 'Chevrolet Corvette ZR1 2012', 95: 'Bentley Arnage Sedan 2009', 96: 'Chevrolet HHR SS 2010', 97: 'Land Rover LR2 SUV 2012', 98: 'Hyundai Azera Sedan 2012', 99: 'Chrysler Aspen SUV 2009', 100: 'Buick Regal GS 2012', 101: 'BMW 3 Series Wagon 2012', 102: 'Jeep Compass SUV 2012', 103: 'Ram C-V Cargo Van Minivan 2012', 104: 'Spyker C8 Convertible 2009', 105: 'Audi S4 Sedan 2007', 106: 'Rolls-Royce Ghost Sedan 2012', 107: 'AM General Hummer SUV 2000', 108: 'Ford Freestar Minivan 2007', 109: 'Bentley Mulsanne Sedan 2011', 110: 'Audi TT Hatchback 2011', 111: 'Mercedes-Benz SL-Class Coupe 2009', 112: 'Chevrolet Silverado 1500 Hybrid Crew Cab 2012', 113: 'Buick Enclave SUV 2012', 114: 'Chevrolet TrailBlazer SS 2009', 115: 'HUMMER H2 SUT Crew Cab 2009', 116: 'McLaren MP4-12C Coupe 2012', 117: 'Dodge Challenger SRT8 2011', 118: 'Suzuki SX4 Hatchback 2012', 119: 'Bugatti Veyron 16.4 Convertible 2009', 120: 'Toyota 4Runner SUV 2012', 121: 'Buick Rainier SUV 2007', 122: 'Chrysler Sebring Convertible 2010', 123: 'Acura Integra Type R 2001', 124: 'Audi V8 Sedan 1994', 125: 'Audi RS 4 Convertible 2008', 126: 'Honda Accord Coupe 2012', 127: 'Audi S4 Sedan 2012', 128: 'Aston Martin Virage Coupe 2012', 129: 'Chevrolet Sonic Sedan 2012', 130: 'Chevrolet Monte Carlo Coupe 2007', 131: 'Volvo XC90 SUV 2007', 132: 'Ford Mustang Convertible 2007', 133: 'Aston Martin Virage Convertible 2012', 134: 'smart fortwo Convertible 2012', 135: 'FIAT 500 Abarth 2012', 136: 'Infiniti G Coupe IPL 2012', 137: 'Dodge Caliber Wagon 2007', 138: 'Hyundai Tucson SUV 2012', 139: 'Acura ZDX Hatchback 2012', 140: 'BMW ActiveHybrid 5 Sedan 2012', 141: 'Ferrari California Convertible 2012', 142: 'Nissan Leaf Hatchback 2012', 143: 'Lamborghini Diablo Coupe 2001', 144: 'Audi S5 Convertible 2012', 145: 'BMW 6 Series Convertible 2007', 146: 'Ferrari 458 Italia Convertible 2012', 147: 'Chevrolet Silverado 2500HD Regular Cab 2012', 148: 'Chevrolet Corvette Convertible 2012', 149: 'Bugatti Veyron 16.4 Coupe 2009', 150: 'Tesla Model S Sedan 2012', 151: 'FIAT 500 Convertible 2012', 152: 'Hyundai Veloster Hatchback 2012', 153: 'Lincoln Town Car Sedan 2011', 154: 'Lamborghini Aventador Coupe 2012', 155: 'Dodge Ram Pickup 3500 Quad Cab 2009', 156: 'Nissan NV Passenger Van 2012', 157: 'Honda Odyssey Minivan 2007', 158: 'Maybach Landaulet Convertible 2012', 159: 'Chevrolet Silverado 1500 Regular Cab 2012', 160: 'Suzuki Kizashi Sedan 2012', 161: 'Chevrolet Tahoe Hybrid SUV 2012', 162: 'Mercedes-Benz Sprinter Van 2012', 163: 'Suzuki Aerio Sedan 2007', 164: 'Audi S5 Coupe 2012', 165: 'Aston Martin V8 Vantage Coupe 2012', 166: 'Chevrolet Malibu Hybrid Sedan 2010', 167: 'Ford F-150 Regular Cab 2012', 168: 'Ford Fiesta Sedan 2012', 169: 'Ford Focus Sedan 2007', 170: 'Bentley Continental Supersports Conv. Convertible 2012', 171: 'Chevrolet Silverado 1500 Classic Extended Cab 2007', 172: 'BMW X5 SUV 2007', 173: 'Jeep Wrangler SUV 2012', 174: 'Acura TL Type-S 2008', 175: 'Chrysler Crossfire Convertible 2008', 176: 'Lamborghini Gallardo LP 570-4 Superleggera 2012', 177: 'Mercedes-Benz E-Class Sedan 2012', 178: 'Chevrolet Express Cargo Van 2007', 179: 'GMC Terrain SUV 2012', 180: 'Dodge Magnum Wagon 2008', 181: 'Honda Accord Sedan 2012', 182: 'Chrysler PT Cruiser Convertible 2008', 183: 'Mazda Tribute SUV 2011', 184: 'BMW M3 Coupe 2012', 185: 'Eagle Talon Hatchback 1998', 186: 'Daewoo Nubira Wagon 2002', 187: 'BMW X6 SUV 2012', 188: 'Lamborghini Reventon Coupe 2008', 189: 'Cadillac SRX SUV 2012', 190: 'MINI Cooper Roadster Convertible 2012', 191: 'Acura RL Sedan 2012', 192: 'BMW 1 Series Convertible 2012', 193: 'Dodge Durango SUV 2007', 194: 'BMW M5 Sedan 2010', 195: 'Chrysler Town and Country Minivan 2012', 196: 'bg'}

model = models.resnet34(pretrained=True)    
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 196)
        
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath ,map_location=torch.device('cpu'))
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    
    # Process a PIL image for use in a PyTorch model

    # Converting image to PIL image using image file path
    pil_im = Image.open(f'{image}' + '.jpg')

    # Building image transform
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    # Transforming image for use with network
    pil_tfd = transform(pil_im)
    
    # Converting to Numpy array 
    array_im_tfd = np.array(pil_tfd)
    
    return array_im_tfd

def predict(image_path, model, topk=5):
    # Implement the code to predict the class from an image file   
    
    # Loading model - using .cpu() for working with CPUs
    loaded_model = load_checkpoint(model).cpu()
    # Pre-processing image
    img = process_image(image_path)
    # Converting to torch tensor from Numpy array
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Adding dimension to image to comply with (B x C x W x H) input of model
    img_add_dim = img_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    loaded_model.eval()
    with torch.no_grad():
        # Running image through network
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Converting probabilities and outputs to lists
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted

classes = []
def find_classes(dir):
    for dir_name in os.listdir(dir):
        if not dir_name.startswith('.'):
            classes.append(dir_name)
            classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
classes, c_to_idx = find_classes('./Car Images/Train Images')

def plot_solution(cardir, model):
    # Testing predict function

    # Inputs are paths to saved model and test image
    model_path = './my_checkpoint1.pth'
    image_path = cardir
    carname = cardir.split('/')[3]
    
    conf2, predicted1 = predict(image_path, model_path, topk=5)
    # Converting classes to names
    names = []
    for i in range(5):
        names += [classes[predicted1[i]]]


    # Creating PIL image
    image = Image.open(image_path+'.jpg')

    # Plotting test image and predicted probabilites
    f, ax = plt.subplots(2,figsize = (6,10))

    ax[0].imshow(image)
    ax[0].set_title(carname)

    y_names = np.arange(len(names))
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    ax[1].barh(y_names, conf2/conf2.sum(), color='darkblue')
    ax[1].set_yticks(y_names)
    ax[1].set_yticklabels(names)
    ax[1].invert_yaxis() 
    
    plt.savefig('./UI_images/'+carname+'_predict'+'.png', bbox_inches="tight")
    plt.close()
    #plt.show()

class PrintLogger(object):  # create file like object

    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end

    def flush(self):  # needed for file like object
        pass


class MainGUI(Tk):

    def __init__(self):
        Tk.__init__(self)
        #self.geometry("500x500")
        self.root = Frame(self)
        self.root.pack()
        self.label_path = Label(self.root, text='Give the path of the image (ex. ./Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543))')
        self.label_path.pack()
        self.label_path = Entry(self.root,width=100)
        self.label_path.focus_set()
        self.label_path.pack()
        self.predict_cls_resnet_button = Button(self.root, text="Predict class of the car (ResNet34)", command=self.predict_cls_resnet)
        self.predict_cls_resnet_button.pack()
        self.predict_img_bbox_resnet_button = Button(self.root, text="Identify the object (ResNet50)", command=self.print_img_bbox_resnet)
        self.predict_img_bbox_resnet_button.pack()
        self.predict_img_bbox_vgg_button = Button(self.root, text="Identify the object (VGG16)", command=self.print_img_bbox_vgg)
        self.predict_img_bbox_vgg_button.pack()
        self.log_widget = ScrolledText(self.root, height=120, width=120, font=("consolas", "8", "normal"))
        self.log_widget.pack()    
        
        
    def load_cls_resnet(self):

        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
        
        model = load_checkpoint('./my_checkpoint1.pth')
        print(model)
        global position
        position = self.log_widget.index(INSERT)
        
        model = torch.nn.DataParallel(model)
        
        return model
    
    def predict_cls_resnet(self):
        
        self.log_widget.delete("1.0","end")
        logger = PrintLogger(self.log_widget)
        logger.textbox.configure(state="disabled")  
        model = self.load_cls_resnet()
        
        cardir=self.label_path.get()#'./Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543'
        plot_solution(cardir, model)
        
        carname = cardir.split('/')[3]
        
        global img_print
        img = Image.open('./UI_images/'+carname+'_predict'+'.png')
        img_print = ImageTk.PhotoImage(img)
        self.log_widget.image_create(position,image=img_print)

    def predict_img_bbox_resnet(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
        
        C = config.Config()
        C.model_path = './'
        C.num_rois = 50
        C.rpn_stride = 16
        from keras_frcnn import resnet as nn
        def format_img_size(img, C):
            """ formats the image size based on config """
            img_min_side = float(C.im_size)
            height = img.shape[0]
            width = img.shape[1]

            if width <= height:
                ratio = img_min_side/width
                new_height = int(img_min_side)#int(ratio * height)
                new_width = int(img_min_side)
            else:
                ratio = img_min_side/height
                new_width = int(img_min_side)#int(ratio * width)
                new_height = int(img_min_side)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return img, ratio

        def format_img_channels(img, C):
            """ formats the image channels based on config """
            img = img[:, :, (2, 1, 0)]
            img = img.astype(np.float32)
            img[:, :, 0] -= C.img_channel_mean[0]
            img[:, :, 1] -= C.img_channel_mean[1]
            img[:, :, 2] -= C.img_channel_mean[2]
            img /= C.img_scaling_factor
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img

        def format_img(img, C):
            """ formats an image for model prediction based on config """
            img, ratio = format_img_size(img, C)
            img = format_img_channels(img, C)
            return img, ratio


        # Method to transform the coordinates of the bounding box to its original size
        def get_real_coordinates(ratio, x1, y1, x2, y2):

            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))

            return (real_x1, real_y1, real_x2 ,real_y2)
        
        num_rois = C.num_rois
        #print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        
        num_features_test = 1024

        input_shape_img_test = (None, None, 3)
        input_shape_features_test = (None, None, num_features_test)

        img_input_test = Input(shape=input_shape_img_test)
        roi_input_test = Input(shape=(C.num_rois, 4))
        feature_map_input_test = Input(shape=input_shape_features_test)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers_test = nn.nn_base(img_input_test)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers_test = nn.rpn(shared_layers_test, num_anchors)

        classifier_test = nn.classifier(feature_map_input_test, roi_input_test, C.num_rois, nb_classes=len(class_mapping))

        model_rpn_test = Model(img_input_test, rpn_layers_test)
        model_classifier_test = Model([feature_map_input_test, roi_input_test], classifier_test)

        print('Loading weights.')
        model_rpn_test.load_weights('./model_rpn_weights_resnet.h5', by_name=True)
        model_classifier_test.load_weights('./model_classifier_weights_resnet.h5', by_name=True)
        
        cardir = './Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543.jpg'
        carname = cardir.split('/')[3]
        
        img = imread(cardir)

        st = time.time()
        if len(img.shape) <3:
            print('Please input an image with 3 channels (RGB)')
        # preprocess image
        X, ratio = format_img(img, C)
        #img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn_test.predict(X)


        R = roi_helpers.rpn_to_roi(Y1, Y2, C, 'tf', overlap_thresh=0.3)
        #print(R.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0]//num_rois + 1):
            ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:,:curr_shape[1],:] = ROIs
                ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
                ROIs = ROIs_padded

            [P_cls,P_regr] = model_classifier_test.predict([F, ROIs])
            #print(P_cls)

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0,ii,:]) < 0.99:# or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x,y,w,h) = ROIs[0,ii,:]

                bboxes[cls_name].append([C.rpn_stride*x,C.rpn_stride*y,C.rpn_stride*(x+w),C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0,ii,:]))

        all_dets = []

        for key in bboxes:
            #print(key)
            #print(len(bboxes[key]))
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.3)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        #print('Elapsed time = {}'.format(time.time() - st))
        #print(all_dets)
        #print(bboxes)
        #if len(bboxes) > 0:
        f, ax = plt.subplots(1,figsize = (10,10))
        ax.grid()
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #plt.show()
        plt.savefig('./UI_images/'+carname+'_predict_bbox'+'.png', bbox_inches="tight")
        plt.close()
        
    def predict_img_bbox_vgg(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
        
        C = config.Config()
        C.model_path = './'
        C.num_rois = 50
        C.rpn_stride = 16
        from keras_frcnn import vgg as nn
        def format_img_size(img, C):
            """ formats the image size based on config """
            img_min_side = float(C.im_size)
            height = img.shape[0]
            width = img.shape[1]

            if width <= height:
                ratio = img_min_side/width
                new_height = int(img_min_side)#int(ratio * height)
                new_width = int(img_min_side)
            else:
                ratio = img_min_side/height
                new_width = int(img_min_side)#int(ratio * width)
                new_height = int(img_min_side)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            return img, ratio

        def format_img_channels(img, C):
            """ formats the image channels based on config """
            img = img[:, :, (2, 1, 0)]
            img = img.astype(np.float32)
            img[:, :, 0] -= C.img_channel_mean[0]
            img[:, :, 1] -= C.img_channel_mean[1]
            img[:, :, 2] -= C.img_channel_mean[2]
            img /= C.img_scaling_factor
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img

        def format_img(img, C):
            """ formats an image for model prediction based on config """
            img, ratio = format_img_size(img, C)
            img = format_img_channels(img, C)
            return img, ratio


        # Method to transform the coordinates of the bounding box to its original size
        def get_real_coordinates(ratio, x1, y1, x2, y2):

            real_x1 = int(round(x1 // ratio))
            real_y1 = int(round(y1 // ratio))
            real_x2 = int(round(x2 // ratio))
            real_y2 = int(round(y2 // ratio))

            return (real_x1, real_y1, real_x2 ,real_y2)
        
        num_rois = C.num_rois
        #print(class_mapping)
        class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
        
        num_features_test = 512

        input_shape_img_test = (None, None, 3)
        input_shape_features_test = (None, None, num_features_test)

        img_input_test = Input(shape=input_shape_img_test)
        roi_input_test = Input(shape=(C.num_rois, 4))
        feature_map_input_test = Input(shape=input_shape_features_test)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers_test = nn.nn_base(img_input_test)

        # define the RPN, built on the base layers
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn_layers_test = nn.rpn(shared_layers_test, num_anchors)

        classifier_test = nn.classifier(feature_map_input_test, roi_input_test, C.num_rois, nb_classes=len(class_mapping))

        model_rpn_test = Model(img_input_test, rpn_layers_test)
        model_classifier_test = Model([feature_map_input_test, roi_input_test], classifier_test)

        print('Loading weights.')
        model_rpn_test.load_weights('./model_rpn_weights_vgg.h5', by_name=True)
        model_classifier_test.load_weights('./model_classifier_weights_vgg.h5', by_name=True)
        
        cardir = './Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543.jpg'
        carname = cardir.split('/')[3]
        
        img = imread(cardir)

        st = time.time()
        if len(img.shape) <3:
            print('Please input an image with 3 channels (RGB)')
        # preprocess image
        X, ratio = format_img(img, C)
        #img_scaled = (np.transpose(X[0,:,:,:],(1,2,0)) + 127.5).astype('uint8')
        X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn_test.predict(X)


        R = roi_helpers.rpn_to_roi(Y1, Y2, C, 'tf', overlap_thresh=0.3)
        #print(R.shape)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}
        for jk in range(R.shape[0]//num_rois + 1):
            ROIs = np.expand_dims(R[num_rois*jk:num_rois*(jk+1),:],axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0]//num_rois:
                #pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0],num_rois,curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:,:curr_shape[1],:] = ROIs
                ROIs_padded[0,curr_shape[1]:,:] = ROIs[0,0,:]
                ROIs = ROIs_padded

            [P_cls,P_regr] = model_classifier_test.predict([F, ROIs])
            #print(P_cls)

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0,ii,:]) < 0.99:# or np.argmax(P_cls[0,ii,:]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0,ii,:])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []
                (x,y,w,h) = ROIs[0,ii,:]

                bboxes[cls_name].append([C.rpn_stride*x,C.rpn_stride*y,C.rpn_stride*(x+w),C.rpn_stride*(y+h)])
                probs[cls_name].append(np.max(P_cls[0,ii,:]))

        all_dets = []

        for key in bboxes:
            #print(key)
            #print(len(bboxes[key]))
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh = 0.3)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk,:]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                all_dets.append((key,100*new_probs[jk]))

                (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                textOrg = (real_x1, real_y1-0)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
                cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

        #print('Elapsed time = {}'.format(time.time() - st))
        #print(all_dets)
        #print(bboxes)
        #if len(bboxes) > 0:
        f, ax = plt.subplots(1,figsize = (10,10))
        ax.grid()
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        #plt.show()
        plt.savefig('./UI_images/'+carname+'_predict_bbox'+'.png', bbox_inches="tight")
        plt.close()
        
    def print_img_bbox_resnet(self):
        self.log_widget.delete("1.0","end")
        logger = PrintLogger(self.log_widget)
        logger.textbox.configure(state="disabled")    
        self.predict_img_bbox_resnet()
        
        cardir=self.label_path.get()#'./Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543'        
        carname = cardir.split('/')[3]
        
        global img_print
        img = Image.open('./UI_images/'+carname+'_predict_bbox'+'.png')
        img_print = ImageTk.PhotoImage(img)
        self.log_widget.image_create(END,image=img_print)
        
    def print_img_bbox_vgg(self):
        self.log_widget.delete("1.0","end")
        logger = PrintLogger(self.log_widget)
        logger.textbox.configure(state="disabled")    
        self.predict_img_bbox_vgg()
        
        cardir=self.label_path.get()#'./Car Images/Test Images/Mercedes-Benz S-Class Sedan 2012/06543'        
        carname = cardir.split('/')[3]
        
        global img_print
        img = Image.open('./UI_images/'+carname+'_predict_bbox'+'.png')
        img_print = ImageTk.PhotoImage(img)
        self.log_widget.image_create(END,image=img_print)
        

if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()