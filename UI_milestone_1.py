# Import required libraries
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


class PrintLogger(object):  # create file like object

    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly

    def flush(self):  # needed for file like object
        pass


class MainGUI(Tk):

    def __init__(self):
        Tk.__init__(self)
        #self.geometry("500x500")
        self.root = Frame(self)
        self.root.pack()
        #self.redirect_button = Button(self.root, text="Redirect console to widget", command=self.redirect_logging)
        #self.redirect_button.pack()
        #self.redirect_button = Button(self.root, text="Redirect console reset", command=self.reset_logging)
        #self.redirect_button.pack()
        self.load_data_button = Button(self.root, text="Load data", command=self.print_load_data)
        self.load_data_button.pack()
        self.map_button = Button(self.root, text="Map classes and annotations", command=self.print_map_data)
        self.map_button.pack()
        self.img_button = Button(self.root, text="Print image", command=self.print_imgs)
        self.img_button.pack()
        self.log_widget = ScrolledText(self.root, height=120, width=120, font=("consolas", "8", "normal"))
        self.log_widget.pack()    

    def reset_logging(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def load_data(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout = logger
        sys.stderr = logger
        
        # Hyperparameters for loading the training, testing data and mapping it with annotations
        validation_split = 0.2
        batch_size = 32
        shuffle = False # This is set to false to mainitain the same order in the training data and the annotations data - explained later
        image_width  = 256
        image_height = 256
        image_size   = (image_width, image_height)
        mask_width   = 28
        mask_height  = 28

        training_data = tf.keras.utils.image_dataset_from_directory(
          directory='./Car Images/Train Images',
          label_mode='categorical',
          validation_split=validation_split,
          subset="training",
          shuffle=shuffle,
          seed=123,
          batch_size=batch_size,
          image_size = image_size)

        validation_data = tf.keras.utils.image_dataset_from_directory(
          directory='./Car Images/Train Images',
          label_mode='categorical',
          validation_split=validation_split,
          subset="validation",
          shuffle=False,
          seed=123,
          batch_size=batch_size,
          image_size = image_size)

        class_labels = training_data.class_names

        return training_data,validation_data,class_labels
    
    def map_class_annot(self):
        
        logger = PrintLogger(self.log_widget)
        sys.stdout = logger
        sys.stderr = logger
        
        training_annotations = pd.read_csv('./Annotations/Train Annotations.csv')
        training_annotations.rename(columns = {'Bounding Box coordinates':'x0', 'Unnamed: 2':'y0','Unnamed: 3':'x1','Unnamed: 4':'y1'}, inplace = True)

        for folder_name in os.listdir('.\Car Images\Train Images'):
            sub_folder_name = os.path.join('.\Car Images\Train Images', folder_name)
            if os.path.isdir(sub_folder_name):
                #print(folder_name)
                for file_name in os.listdir(sub_folder_name):
                    file_path_name = os.path.join(sub_folder_name, file_name)
                    if os.path.isfile(file_path_name):
                        training_annotations['Image class'][training_annotations['Image Name']==file_name] = str(folder_name)
                        training_annotations['Image Name'][training_annotations['Image Name']==file_name] = str(file_path_name)
        
        i = 0
        for file_name in training_annotations['Image Name']:
            if i<100:
                img_original = imread(file_name)
                fig,ax = plt.subplots(1)
                ax.imshow(img_original)
                mask_data = training_annotations[training_annotations['Image Name']==file_name]
                mask = patches.Rectangle((mask_data['x0'].values[0], mask_data['y0'].values[0]), mask_data['x1'].values[0]-mask_data['x0'].values[0], mask_data['y1'].values[0]-mask_data['y0'].values[0], linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(mask)
                #plt.show()
                plt.savefig('./UI_images/'+file_name[-9:-4]+'.png')
                plt.close()
                i+=1

        return training_annotations 
    
    def print_load_data(self):
        
        training_data,validation_data,class_labels = self.load_data()
        
        print('Total number of batches (of size 32) in the training data:',len(training_data),'\n','Shape and type of each batch in the training data:', training_data,'\n','Number of classes in training dataset:',np.size(training_data.class_names),'\n','Total number of batches (of size 32) in the validation data:', len(validation_data),'Shape and type of each batch in the validation data:', validation_data,'\n','Number of classes in validation dataset:',np.size(validation_data.class_names),'\n','Names of the classes in the dataset:',class_labels)
        
    def print_map_data(self):
        training_annotations = self.map_class_annot()
        print('Mapped data:',training_annotations.head(),training_annotations.info)
        global position
        position = self.log_widget.index(INSERT)
        
    def print_imgs(self):
        global img_print
        img_no = randint(1, 100)
        if img_no <10:
            file_name = '0000'+str(img_no)+'.png'
        elif img_no <100:
            file_name = '000'+str(img_no)+'.png'
        else:
            file_name = '00'+str(img_no)+'.png'
        
        img = Image.open('./UI_images/'+file_name)
        img_print = ImageTk.PhotoImage(img)
        self.log_widget.image_create(position,image=img_print)

if __name__ == "__main__":
    app = MainGUI()
    app.mainloop()