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

def print_get_data():     
    
    found_bg = False
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    all_data = []
    visualise = True
    
    f = open('./Annotations/Train Annotations_temp.csv')
        
    reader_obj = csv.reader(f)
    
    for row in reader_obj:
        
        filename,x1,y1,x2,y2,class_name = row

        if class_name not in classes_count:
            classes_count[class_name] = 1
        else:
            classes_count[class_name] += 1

        if class_name not in class_mapping:
            if class_name == 'bg' and found_bg == False:
                print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                found_bg = True
            class_mapping[class_name] = len(class_mapping)

        if filename not in all_imgs:
            all_imgs[filename] = {}

            img = cv2.imread(filename)
            (rows,cols) = img.shape[:2]

            all_imgs[filename]['filepath'] = filename
            all_imgs[filename]['width'] = cols
            all_imgs[filename]['height'] = rows
            all_imgs[filename]['bboxes'] = []
            all_imgs[filename]['imageset'] = 'trainval'

        all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

    f.close()
    #print('i am out')
    for key in all_imgs:
        all_data.append(all_imgs[key])    
    
    print('Parsed annotation files')
    print('Sample data:',all_data[0])
        
    return all_data, classes_count, class_mapping

def print_start_training():
    print('Starting training...')
        
def print_epoch_data(C,num_epochs,epoch_num,mean_overlapping_bboxes, epoch_length,loss_vals,iter_num,selected_pos_samples,class_acc,loss_rpn_cls,loss_rpn_regr,loss_class_cls,loss_class_regr,time,start_time,curr_loss,best_loss):
    print('IN training...')
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
    
    if mean_overlapping_bboxes == 0:
        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')
    else:
        print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
        
    progbar.update(iter_num, [('rpn_cls', np.mean(loss_vals[:iter_num, 0])), ('rpn_regr', np.mean(loss_vals[:iter_num, 1])),
                                              ('detector_cls', np.mean(loss_vals[:iter_num, 2])), ('detector_regr', np.mean(loss_vals[:iter_num, 3])),
                                             ("average number of objects", len(selected_pos_samples))])
    if C.verbose:
        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
        print('Loss RPN regression: {}'.format(loss_rpn_regr))
        print('Loss Detector classifier: {}'.format(loss_class_cls))
        print('Loss Detector regression: {}'.format(loss_class_regr))
        print('Elapsed time: {}'.format(time.time() - start_time))   
        
    if curr_loss < best_loss:
        if C.verbose:
            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))


class PrintLogger(object):  # create file like object

    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.update_idletasks()
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly

    def flush(self):  # needed for file like object
        pass

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def rpn_loss_regr_fixed_num(y_true, y_pred):

        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)

        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])

    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    """Loss function for rpn classification
    Args:
        num_anchors: number of anchors (9 in here)
        y_true[:, :, :, :9]: [0,1,0,0,0,0,0,1,0] means only the second and the eighth box is valid which contains pos or neg anchor => isValid
        y_true[:, :, :, 9:]: [0,1,0,0,0,0,0,0,0] means the second box is pos and eighth box is negative
    Returns:
        lambda * sum((binary_crossentropy(isValid*y_pred,y_true))) / N
    """
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])

    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    """Loss function for rpn regression
    Args:
        num_anchors: number of anchors (9 in here)
    Returns:
        Smooth L1 loss function 
                           0.5*x*x (if x_abs < 1)
                           x_abx - 0.5 (otherwise)
    """
    def class_loss_regr_fixed_num(y_true, y_pred):
        #print(y_pred.dtype)
        #print(y_true.dtype)
        y_true = tf.cast(y_true, tf.float32)
        #print(y_true.dtype)
        x = y_true[:, :, 4*int(num_classes):] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(y_true[:, :, :4*int(num_classes)] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*int(num_classes)])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

def validation(model, validloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    # change model to work with cuda
    model.to('cuda')

    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):
    
        # Change images and labels to work with cuda
        images, labels = images.to('cuda'), labels.to('cuda')

        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

class MainGUI(Tk):

    def __init__(self):
        Tk.__init__(self)
        #self.geometry("500x500")
        self.root = Frame(self)
        self.root.pack()
        #self.label_no_imgs = Label(self.root, text='Number of training images (less than or equal to 8,144)')
        #self.label_no_imgs.pack()
        #self.entry_no_imgs = Entry(self.root,width=50)
        #self.entry_no_imgs.focus_set()
        #self.entry_no_imgs.pack()
        self.label_no_epochs = Label(self.root, text='Number of epochs')
        self.label_no_epochs.pack()
        self.entry_no_epochs = Entry(self.root,width=10)
        self.entry_no_epochs.focus_set()
        self.entry_no_epochs.pack()
        self.train_train_model_clf = Button(self.root, text='Train image classifier with ResNet34', command=self.print_train_model_clf)
        self.train_train_model_clf.pack()
        self.train_model_obj_det_vgg = Button(self.root, text='Train Faster RCNN with VGG16', command=self.print_train_model_obj_det_vgg)
        self.train_model_obj_det_vgg.pack()
        self.train_model_obj_det_resnet = Button(self.root, text='Train Faster RCNN with ResNet50', command=self.print_train_model_obj_det_resnet)
        self.train_model_obj_det_resnet.pack()
        self.log_widget = ScrolledText(self.root, height=120, width=120, font=("consolas", "8", "normal"))
        self.log_widget.pack()  

    def reset_logging(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def print_train_model_obj_det_resnet(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
        
        #self.no_imgs = self.entry_no_imgs.get()
        self.num_epochs = self.entry_no_epochs.get()
        
        C = config.Config()
        C.model_path = './'
        C.num_rois = 50
        C.rpn_stride = 16
        
        from keras_frcnn import resnet as nn

        all_imgs,classes_count,class_mapping = print_get_data()#int(self.no_imgs)
        print_start_training()
        
        if 'bg' not in classes_count:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)

        C.class_mapping = class_mapping
        
        train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']        
        
        data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, 'tf', mode='train')
        
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        shared_layers = nn.nn_base(img_input, trainable=False)
        C.base_net_weights = './resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        
        optimizer = Adam(learning_rate=1e-4, clipnorm=0.001)
        optimizer_classifier = Adam(learning_rate=1e-4, clipnorm=0.001)
        
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn = nn.rpn(shared_layers, num_anchors)
        
        classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)

        model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
        
        model_all.compile(optimizer='sgd', loss='mae')

        epoch_length = int(100)
        num_epochs = int(self.num_epochs)
        iter_num = 0

        rpn_weight_path = None

        loss_vals = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        start_time = time.time()

        best_loss = np.Inf

        class_mapping_inv = {v: k for k, v in class_mapping.items()}

        vis = True

        for epoch_num in range(num_epochs):

            # first 3 epoch is warmup
            if epoch_num < 2 and rpn_weight_path is not None:
                K.set_value(model_rpn.optimizer.lr, 1e-4)
                K.set_value(model_classifier.optimizer.lr, 1e-4)

            while True:
                try:
                    X, Y, img_data = next(data_gen_train)
                    #print(len(img_data))
                    if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                    loss_rpn = model_rpn.train_on_batch(X, Y)

                    P_rpn = model_rpn.predict_on_batch(X)
                    #print(P_rpn)
                    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', use_regr=True, overlap_thresh=0.4, max_boxes=300)
                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                    if X2 is None:
                        rpn_accuracy_rpn_monitor.append(0)
                        rpn_accuracy_for_epoch.append(0)
                        continue

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)

                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))

                    if C.num_rois > 1:
                       
                        if len(pos_samples) < C.num_rois//2:
                            selected_pos_samples = pos_samples.tolist()
                        else:
                            selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                        try:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                        except:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        selected_pos_samples = pos_samples.tolist()
                        selected_neg_samples = neg_samples.tolist()
                        if np.random.randint(0, 2):
                                        sel_samples = random.choice(neg_samples)
                        else:
                                        sel_samples = random.choice(pos_samples)

                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    loss_vals[iter_num, 0] = loss_rpn[1]
                    loss_vals[iter_num, 1] = loss_rpn[2]

                    loss_vals[iter_num, 2] = loss_class[1]
                    loss_vals[iter_num, 3] = loss_class[2]
                    loss_vals[iter_num, 4] = loss_class[3]

                    iter_num += 1

                    if iter_num == epoch_length:
                        loss_rpn_cls = np.mean(loss_vals[:, 0])
                        loss_rpn_regr = np.mean(loss_vals[:, 1])
                        loss_class_cls = np.mean(loss_vals[:, 2])
                        loss_class_regr = np.mean(loss_vals[:, 3])
                        class_acc = np.mean(loss_vals[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            best_loss = curr_loss
                            model_all.save_weights(C.model_path)
                            
                            print_epoch_data(C,num_epochs,epoch_num,mean_overlapping_bboxes, epoch_length,loss_vals,iter_num,selected_pos_samples,class_acc,loss_rpn_cls,loss_rpn_regr,loss_class_cls,loss_class_regr,time,start_time,curr_loss,best_loss)
    
                        break
                    
                except Exception as e:
                    print('Exception 123: {}'.format(e))
                    continue

        print('Training complete, saving the model.')
        
        model_rpn.save('./model_rpn_resnet_UI.h5')
        model_classifier.save('./model_classifier_resnet_UI.h5')
        model_all.save('./model_all_resnet_UI.h5')
        
    def print_train_model_obj_det_vgg(self):
        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
            
        #self.no_imgs = self.entry_no_imgs.get()
        self.num_epochs = self.entry_no_epochs.get()
        
        C = config.Config()
        C.model_path = './'
        C.num_rois = 50
        C.rpn_stride = 16
        from keras_frcnn import vgg as nn
        all_imgs,classes_count,class_mapping = print_get_data()#int(self.no_imgs)
        print_start_training()
        
        if 'bg' not in classes_count:
            classes_count['bg'] = 0
            class_mapping['bg'] = len(class_mapping)
        
        C.class_mapping = class_mapping
        
        train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']        
        
        data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, 'tf', mode='train')
        
        img_input = Input(shape=(None, None, 3))
        roi_input = Input(shape=(None, 4))
        shared_layers = nn.nn_base(img_input, trainable=False)
        C.base_net_weights = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
        
        optimizer = Adam(learning_rate=1e-4, clipnorm=0.001)
        optimizer_classifier = Adam(learning_rate=1e-4, clipnorm=0.001)
        
        num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
        rpn = nn.rpn(shared_layers, num_anchors)
        
        classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

        model_rpn = Model(img_input, rpn[:2])
        model_classifier = Model([img_input, roi_input], classifier)
        model_all = Model([img_input, roi_input], rpn[:2] + classifier)

        model_rpn.load_weights(C.base_net_weights, by_name=True)
        model_classifier.load_weights(C.base_net_weights, by_name=True)

        model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
        model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)], metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
        
        model_all.compile(optimizer='sgd', loss='mae')

        epoch_length = int(100)
        num_epochs = int(self.num_epochs)
        iter_num = 0

        rpn_weight_path = None

        loss_vals = np.zeros((epoch_length, 5))
        rpn_accuracy_rpn_monitor = []
        rpn_accuracy_for_epoch = []
        start_time = time.time()

        best_loss = np.Inf

        class_mapping_inv = {v: k for k, v in class_mapping.items()}

        vis = True

        for epoch_num in range(num_epochs):

            # first 3 epoch is warmup
            if epoch_num < 2 and rpn_weight_path is not None:
                K.set_value(model_rpn.optimizer.lr, 1e-4)
                K.set_value(model_classifier.optimizer.lr, 1e-4)

            while True:
                try:
                    X, Y, img_data = next(data_gen_train)
                    #print(len(img_data))
                    if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                        mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                        rpn_accuracy_rpn_monitor = []
                    loss_rpn = model_rpn.train_on_batch(X, Y)

                    P_rpn = model_rpn.predict_on_batch(X)
                    #print(P_rpn)
                    R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, 'tf', use_regr=True, overlap_thresh=0.4, max_boxes=300)
                    # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                    X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

                    if X2 is None:
                        rpn_accuracy_rpn_monitor.append(0)
                        rpn_accuracy_for_epoch.append(0)
                        continue

                    neg_samples = np.where(Y1[0, :, -1] == 1)
                    pos_samples = np.where(Y1[0, :, -1] == 0)

                    if len(neg_samples) > 0:
                        neg_samples = neg_samples[0]
                    else:
                        neg_samples = []

                    if len(pos_samples) > 0:
                        pos_samples = pos_samples[0]
                    else:
                        pos_samples = []

                    rpn_accuracy_rpn_monitor.append(len(pos_samples))
                    rpn_accuracy_for_epoch.append((len(pos_samples)))

                    if C.num_rois > 1:
                       
                        if len(pos_samples) < C.num_rois//2:
                            selected_pos_samples = pos_samples.tolist()
                        else:
                            selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                        try:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                        except:
                            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                        sel_samples = selected_pos_samples + selected_neg_samples
                    else:
                        # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                        selected_pos_samples = pos_samples.tolist()
                        selected_neg_samples = neg_samples.tolist()
                        if np.random.randint(0, 2):
                                        sel_samples = random.choice(neg_samples)
                        else:
                                        sel_samples = random.choice(pos_samples)

                    loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

                    loss_vals[iter_num, 0] = loss_rpn[1]
                    loss_vals[iter_num, 1] = loss_rpn[2]

                    loss_vals[iter_num, 2] = loss_class[1]
                    loss_vals[iter_num, 3] = loss_class[2]
                    loss_vals[iter_num, 4] = loss_class[3]

                    iter_num += 1

                    if iter_num == epoch_length:
                        loss_rpn_cls = np.mean(loss_vals[:, 0])
                        loss_rpn_regr = np.mean(loss_vals[:, 1])
                        loss_class_cls = np.mean(loss_vals[:, 2])
                        loss_class_regr = np.mean(loss_vals[:, 3])
                        class_acc = np.mean(loss_vals[:, 4])

                        mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                        rpn_accuracy_for_epoch = []

                        curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                        iter_num = 0
                        start_time = time.time()

                        if curr_loss < best_loss:
                            best_loss = curr_loss
                            model_all.save_weights(C.model_path)
                            
                            print_epoch_data(C,num_epochs,epoch_num,mean_overlapping_bboxes, epoch_length,loss_vals,iter_num,selected_pos_samples,class_acc,loss_rpn_cls,loss_rpn_regr,loss_class_cls,loss_class_regr,time,start_time,curr_loss,best_loss)
    
                        break
                    
                except Exception as e:
                    print('Exception 123: {}'.format(e))
                    continue

        print('Training complete, saving the model.')
        
        model_rpn.save('./model_rpn_vgg_UI.h5')
        model_classifier.save('./model_classifier_vgg_UI.h5')
        model_all.save('./model_all_vgg_UI.h5')

        
    def print_train_model_clf(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = logger
        sys.stderr = logger
        
        print('Pytorch version: ',torch.__version__)
        print('CUDA version: ',torch.version.cuda)
        
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            print('CUDA is available, Training on GPU ...')
        else:
            print('CUDA is not available!  Training on CPU ...')
            
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        
        data_dir = './Car Images/'
        train_dir = data_dir + '/Train Images'
        test_dir = data_dir + '/Test Images'
        
        #self.no_imgs = self.entry_no_imgs.get()
        self.num_epochs = self.entry_no_epochs.get()
        
        # Training transform includes random rotation and flip to build a more robust model
        train_transforms = transforms.Compose([transforms.Resize((244,244)),
                                               transforms.RandomRotation(30),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        validation_transforms = transforms.Compose([transforms.Resize((244,244)),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Load the datasets with ImageFolder
        train_data = datasets.ImageFolder(data_dir + '/Train Images', transform=train_transforms)
        valid_data = datasets.ImageFolder(data_dir + '/Train Images', transform=validation_transforms)

        # Using the image datasets and the trainforms, define the dataloaders
        # The trainloader will have shuffle=True so that the order of the images do not affect the model
        trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
        validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
        
        model = models.resnet34(pretrained=True)    
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 196)
        print(model.fc)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
        
        epochs = int(self.num_epochs)
        steps = 0
        print_every = 40

        # change to gpu mode
        model.to('cuda')
        model.train()
        for e in range(epochs):
            print('Starting Training...')
            running_loss = 0

            # Iterating over data to carry out training step
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                # zeroing parameter gradients
                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Carrying out validation step
                if steps % print_every == 0:
                    # setting model to evaluation mode during validation
                    model.eval()

                    # Gradients are turned off as no longer in training
                    with torch.no_grad():
                        valid_loss, accuracy = validation(model, validloader, criterion)

                    print(f"No. epochs: {e+1}, \
                    Training Loss: {round(running_loss/print_every,3)} \
                    Valid Loss: {round(valid_loss/len(validloader),3)} \
                    Valid Accuracy: {round(float(accuracy/len(validloader)),3)}")


                    # Turning training back on
                    model.train()
                    lrscheduler.step(accuracy * 100)
                    
        # Saving: feature weights, new model.fc, index-to-class mapping, optimiser state, and No. of epochs
        checkpoint = {'state_dict': model.state_dict(),
                      'model': model.fc,
                      'class_to_idx': train_data.class_to_idx,
                      'opt_state': optimizer.state_dict,
                      'num_epochs': epochs}

        torch.save(checkpoint, './clf_resnet_UI.pth')
            
            
if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()