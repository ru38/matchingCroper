#-*- coding:utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtWidgets, uic , QtGui
import sys
import cv2
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #not use gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #log level, {'0', '1', '2'}

import math
import shutil
import random
import threading

from PIL.ImageQt import ImageQt
from PIL import Image
from PIL import ImageOps
from PIL import ImageFile

import scipy.misc
from scipy.misc import toimage
import json
from collections import OrderedDict
import tkinter
import tkinter.font
from tkinter import *
import tkinter.filedialog
from skimage.measure import compare_ssim
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import csv
import time

patternList = [cv2.TM_CCOEFF,cv2.TM_CCOEFF_NORMED,cv2.TM_CCORR,cv2.TM_CCORR_NORMED,cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]
patternList2 = ['TM_CCOEFF','TM_CCOEFF_NORMED','TM_CCORR','TM_CCORR_NORMED','TM_SQDIFF','TM_SQDIFF_NORMED']
thresholdList =[cv2.THRESH_BINARY,cv2.THRESH_BINARY_INV,cv2.THRESH_TRUNC,cv2.THRESH_TOZERO,cv2.THRESH_TOZERO_INV,cv2.THRESH_TRIANGLE]
thresholdList2 =["THRESH_BINARY","THRESH_BINARY_INV","THRESH_TRUNC","THRESH_TOZERO","THRESH_TOZERO_INV","THRESH_TRIANGLE"]
filterList =[cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.ADAPTIVE_THRESH_MEAN_C]
filterList2 =["GAUSSIAN","MEAN"]


class MyMplCanvas(FigureCanvas):
  def __init__(self, parent=None, width=5, height=3, dpi=100):
      self.fig = Figure(figsize=(width, height), dpi=dpi)
      self.axes = self.fig.add_subplot(111)

      FigureCanvas.__init__(self, self.fig)
      self.setParent(parent)

      FigureCanvas.setSizePolicy(self,QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
      FigureCanvas.updateGeometry(self)

class Main(QMainWindow):
   def __init__(self):
      super().__init__()   
      self.formMain=uic.loadUi('ui_main.ui',self)
      self.widgetImg1 = QHBoxLayout(self.lb_Histogram)  
      self.histoWidget = MyMplCanvas(self.lb_Histogram)  
      self.widgetImg1.addWidget(self.histoWidget)   
      self.histoWidget.axes.text(0, 1.03, "Histogram", fontsize=13, fontweight='bold',transform=self.histoWidget.axes.transAxes) 

      self.setAcceptDrops(True)
      self.setMouseTracking(True)
      self.lb_imgHisto.setAcceptDrops(True)
      self.lb_imgScale.setAcceptDrops(True)
      self.lb_imgThreshold.setAcceptDrops(True)
      self.lb_mask.setAcceptDrops(True)
      self.lb_img1.setAcceptDrops(True)
      self.lb_img2.setAcceptDrops(True)

      ## Degree Adjust ## 
      self.btn_open1.clicked.connect(self.openDirectory0)
      self.btn_open3.clicked.connect(self.openDirectoryALL)
      self.btn_openTH.clicked.connect(self.openDirectoryTH)
      self.actionOpenDir_2.triggered.connect(self.openMainDir)
      self.actionExit.triggered.connect(self.exit)

      self.btn_doAll.clicked.connect(self.doAll)
      self.btn_doAll4.clicked.connect(self.doAll2)
      self.btn_doAll9.clicked.connect(self.doAll3)
      self.btn_doOnly9.clicked.connect(self.doOnlyCrop9)

      self.btn_pattrenMatch.clicked.connect(lambda:self.pattrenMatch(num=1,path = self.te_path1.toPlainText()))
      self.btn_rotate.clicked.connect(lambda:self.rotate(num=1,path=self.te_path1.toPlainText()))
      self.btn_findDegree.clicked.connect(self.findDegree)
      self.btn_pattrenMatch_2.clicked.connect(lambda:self.pattrenMatch(num=2,path=None))
      self.btn_rotate_2.clicked.connect(lambda:self.rotate(num=2, path=None))
      self.btn_findDegree_2.clicked.connect(self.findDegree)

      self.btn_roiSet.clicked.connect(self.roiSet)
      self.btn_roiCut.clicked.connect(self.roiCut)
      self.btn_roiSet_2.clicked.connect(self.roiSet2)
      self.btn_roiCut_2.clicked.connect(lambda:self.roiCut2(num=0,path=None))
      self.btn_roiSet_3.clicked.connect(self.roiSet3)
      self.btn_roiCut_3.clicked.connect(lambda:self.roiCut3(num=0,path=None))
      self.btn_roiSet_4.clicked.connect(self.roiSet4)

      self.btn_save.clicked.connect(lambda:self.save(num=1,path=self.te_path1.toPlainText()))
      self.btn_roiSave.clicked.connect(self.roiSave)
      self.btn_roiLoad.clicked.connect(lambda:self.roiload(mode=1))
      self.roiload(mode=0)
      self.btn_viewBig.clicked.connect(self.viewBig)

      ## absdiff, MatchRate, SSIM ## 
      self.btn_openlistCal.clicked.connect(self.openAllCal)#for subtract.. etc..
      self.btn_doallCal.clicked.connect(self.doAllCal) #for automatic subtract
      self.cb_pattrenMatch.clear()
      self.cb_pattrenMatch.addItems(patternList2)
      self.cb_pattrenMatch.setCurrentIndex(1)
      self.btn_add.clicked.connect(self.add)
      self.btn_sub.clicked.connect(self.sub)
      self.btn_mul.clicked.connect(self.mul)
      self.btn_div.clicked.connect(self.div)
      self.btn_absDiff.clicked.connect(self.absdiff)
      self.btn_bitand.clicked.connect(self.bitand)
      self.btn_bitor.clicked.connect(self.bitor)
      self.btn_bitnot.clicked.connect(self.bitnot)
      self.btn_bitxor.clicked.connect(self.bitxor)

      ## Histogram ## 
      self.btn_openlistHisto.clicked.connect(self.openAllHisto)#for Histogram
      self.btn_doallHisto.clicked.connect(lambda:self.doAllHisto(self.histNum1.value(),self.histNum2.value()))
      self.btn_doallHisto_2.clicked.connect(lambda:self.doAllHisto(self.histNum1_2.value(),self.histNum2_2.value()))
      self.btn_histogram.clicked.connect(self.histogramTest)

      ## Bright, Scale Change ##
      self.btn_openlistBright.clicked.connect(self.openAllBright)
      self.btn_doallBright.clicked.connect(self.doAllBright)
      self.btn_doallScale.clicked.connect(self.doAllScale)
      self.btn_scaleTest.clicked.connect(self.scaleTest)

      ## threshold ##
      self.filterDoList=[]
      self.filterDoListModel = QtGui.QStandardItemModel()
      self.btn_openlistThreshold.clicked.connect(self.openAllThreshold)
      self.btn_threshold.clicked.connect(self.threshold)
      self.btn_filter.clicked.connect(self.filter)
      self.btn_morpOpen.clicked.connect(self.morpOpen)
      self.btn_morpClose.clicked.connect(self.morpClose)
      self.btn_morpErosion.clicked.connect(self.morpErosion)
      self.btn_morpDilation.clicked.connect(self.morpDilation)
      self.btn_morpGRADIENT.clicked.connect(self.morpGRADIENT)
      self.btn_morpTOPHAT.clicked.connect(self.morpTOPHAT)
      self.btn_morpBLACKHAT.clicked.connect(self.morpBLACKHAT)
      self.btn_morpHITMISS.clicked.connect(self.morpHITMISS)
      self.btn_resetThreshold.clicked.connect(self.resetThreshold)   
      self.btn_mask.clicked.connect(self.mask)
      self.btn_colorReverse.clicked.connect(self.colorReverse)
      self.btn_equalizeHist.clicked.connect(self.equalizeHist)
      self.btn_sharpening.clicked.connect(self.sharpening)

      self.cb_filter.clear()
      self.cb_filter.addItems(filterList2)
      self.cb_threshold.clear()
      self.cb_threshold.addItems(thresholdList2)
      self.btn_resetDoList.clicked.connect(self.resetDoList)

      self.btn_doallThreshold.clicked.connect(self.doAllThreshold) 
      self.rBtn_addListThreshold.clicked.connect(lambda:self.addToDoList("Threshold"))
      self.rBtn_addListFilter.clicked.connect(lambda:self.addToDoList("Filter"))
      self.rBtn_addListMask.clicked.connect(lambda:self.addToDoList("Mask"))
      self.rBtn_addListOpen.clicked.connect(lambda:self.addToDoList("Open"))
      self.rBtn_addListClose.clicked.connect(lambda:self.addToDoList("Close"))
      self.rBtn_addListErosion.clicked.connect(lambda:self.addToDoList("Erosion"))
      self.rBtn_addListDilation.clicked.connect(lambda:self.addToDoList("Dilation"))
      self.rBtn_addListGRADIENT.clicked.connect(lambda:self.addToDoList("GRADIENT"))
      self.rBtn_addListColorReverse.clicked.connect(lambda:self.addToDoList("ColorReverse"))
      self.rBtn_addListEqualizeHist.clicked.connect(lambda:self.addToDoList("EqualizeHist"))
      self.rBtn_addListSharpening.clicked.connect(lambda:self.addToDoList("Sharpening"))

      self.show()

   def exit(self):
      self.close()

   def dragEnterEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()
   def dropEvent(self, e):
        if e.mimeData().hasUrls:
            e.accept()
            for url in e.mimeData().urls():
                fname = str(url.toLocalFile())
                if self.lb_imgHisto.underMouse():
                  self.te_pathHistoTest.setText(fname)
                  self.lb_imgHisto.setPixmap(QtGui.QPixmap(fname))
                  self.lb_imgHisto.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                  self.lb_imgHisto.setScaledContents(True)
                if self.lb_imgScale.underMouse():
                  self.te_pathScaleTest.setText(fname)
                  self.lb_imgScale.setPixmap(QtGui.QPixmap(fname))
                  self.lb_imgScale.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                  self.lb_imgScale.setScaledContents(True)
                if self.lb_imgThreshold.underMouse():
                  self.te_pathThreshold.setText(fname)
                  self.imgThreshold = cv2.imread(self.te_pathThreshold.toPlainText(),cv2.IMREAD_GRAYSCALE)
                  self.lb_imgThreshold.setPixmap(QtGui.QPixmap(fname))
                  self.lb_imgThreshold.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                  self.lb_imgThreshold.setScaledContents(True)
                if self.lb_mask.underMouse():
                  self.lb_mask.setText(fname)   
                if self.lb_img1_2.underMouse():
                  self.te_path1_2.setText(fname)
                  self.lb_img1_2.setPixmap(QtGui.QPixmap(fname))
                  self.lb_img1_2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                  self.lb_img1_2.setScaledContents(True)      
                if self.lb_img2_2.underMouse():
                  self.te_path2_2.setText(fname)
                  self.lb_img2_2.setPixmap(QtGui.QPixmap(fname))
                  self.lb_img2_2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                  self.lb_img2_2.setScaledContents(True)

  
        else:
            e.ignore()

   def resetDoList(self):
        self.filterDoList=[]
        self.filterDoListModel = QtGui.QStandardItemModel()
        self.lv_listTH_2.setModel( self.filterDoListModel)
    
   def viewBig(self):
      try:
         cv2.imshow("img",self.dstBig)
      except Exception as ex:
         QMessageBox.information(self, 'viewBig', str(ex), QMessageBox.Ok )

   def resetThreshold(self):
         try:
            self.imgThreshold = cv2.imread(self.te_pathThreshold.toPlainText(),cv2.IMREAD_GRAYSCALE)
            #self.lb_imgThreshold.setPixmap(QtGui.QPixmap.fromImage(scipy.misc.toimage(self.imgThreshold)))
         except Exception as ex:
            print(ex)

   def morpOpen(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_open.value(), self.spinBox_open.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpClose(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_close.value(), self.spinBox_close.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpErosion(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_erosion.value(), self.spinBox_erosion.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpDilation(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_dilation.value(), self.spinBox_dilation.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpGRADIENT(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_GRADIENT.value(), self.spinBox_GRADIENT.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpTOPHAT(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_TOPHAT.value(), self.spinBox_TOPHAT.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpBLACKHAT(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_BLACKHAT.value(), self.spinBox_BLACKHAT.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def morpHITMISS(self):
         try:
            img = self.imgThreshold
            kernel = np.ones((self.spinBox_HITMISS.value(), self.spinBox_HITMISS.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def mask(self):
         try:
            img = self.imgThreshold
            maskimg = cv2.imread(self.lb_mask.text(), cv2.IMREAD_GRAYSCALE)
            masked = cv2.bitwise_or(img,img,mask=maskimg) 
            self.imgThreshold=masked
            cv2.imshow("Result", masked);
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def colorReverse(self):
         try:
            img = self.imgThreshold
            result = cv2.bitwise_not(img)
            self.imgThreshold=result
            cv2.imshow("Result", result);
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def threshold(self):
         try:
            img = self.imgThreshold
            ret, result = cv2.threshold(img,self.spinBox.value(),self.spinBox_2.value(), thresholdList[self.cb_threshold.currentIndex()])
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def filter(self):    
         try:
            img = self.imgThreshold
            result = cv2.adaptiveThreshold(img, self.spinBox_5.value(), filterList[self.cb_filter.currentIndex()], thresholdList[self.cb_threshold.currentIndex()], self.spinBox_3.value(), self.spinBox_4.value())   
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def equalizeHist(self):    
         try:
            img = self.imgThreshold
            result = cv2.equalizeHist(img)   
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def sharpening(self):    
         try:
            img = self.imgThreshold 
            sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            #sharpening_1 = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1],[-1, 2, 9, 2, -1], [-1, 2, 2, 2, -1],[-1, -1, -1, -1, -1]]) / 9.0
            result = cv2.filter2D(img, -1, sharpening_1)
            self.imgThreshold=result
            cv2.imshow("Result", result)
         except Exception as ex:
            print(ex)
            QMessageBox.information(self, 'ex', str(sys.exc_info()), QMessageBox.Ok )

   def sharpening2(self,img):
         try:
            sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(img, -1, sharpening_1)
         except Exception as ex:
            print(ex)
         return result;

   def equalizeHist2(self,img):
         try:
            result = cv2.equalizeHist(img)
         except Exception as ex:
            print(ex)
         return result;

   def colorReverse2(self,img):
         try:
            result = cv2.bitwise_not(img)
         except Exception as ex:
            print(ex)
         return result;

   def morpErosion2(self,img):
         try:
            kernel = np.ones((self.spinBox_erosion.value(), self.spinBox_erosion.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
         except Exception as ex:
            print(ex)
         return result;

   def morpDilation2(self,img):
         try:
            kernel = np.ones((self.spinBox_dilation.value(), self.spinBox_dilation.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
         except Exception as ex:
            print(ex)
         return result;

   def morpGRADIENT2(self,img):
         try:
            kernel = np.ones((self.spinBox_GRADIENT.value(), self.spinBox_GRADIENT.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
         except Exception as ex:
            print(ex)
         return result;

   def mask2(self,img):
         try:
            maskimg = cv2.imread(self.lb_mask.text(), cv2.IMREAD_GRAYSCALE)
            masked = cv2.bitwise_or(img,img,mask=maskimg) 
         except Exception as ex:
            print(ex)
         return masked;

   def morpOpen2(self,img):
         try:
            kernel = np.ones((self.spinBox_open.value(), self.spinBox_open.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
         except Exception as ex:
            print(ex)
         return result;

   def morpClose2(self,img):
         try:
            kernel = np.ones((self.spinBox_close.value(), self.spinBox_close.value()), np.uint8)
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
         except Exception as ex:
            print(ex)
         return result;

   def threshold2(self,img):
         try:
            ret, result = cv2.threshold(img,self.spinBox.value(),self.spinBox_2.value(), thresholdList[self.cb_threshold.currentIndex()])
         except Exception as ex:
            print(ex)
         return result;

   def filter2(self,img):    
         try:
            result = cv2.adaptiveThreshold(img, self.spinBox_5.value(), filterList[self.cb_filter.currentIndex()], thresholdList[self.cb_threshold.currentIndex()], self.spinBox_3.value(), self.spinBox_4.value())   
         except Exception as ex:
            print(ex)
         return result;


   def openAllCal(self):      
      dirPath=''
      dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(dirPath)
      self.te_dirCal.setText(dirPath)
      if dirPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         try:
            self.fullFileList=[]
            self.fullFileList2=[]

            model1 = QtGui.QStandardItemModel()
            model2 = QtGui.QStandardItemModel()

            with open(dirPath+'/output_result.csv','r') as f:
               reader = csv.DictReader(f)
               for txt in reader :
                  for k,v in txt.items():
                    # print(k)
                     if k == 'x:image':
                        self.fullFileList.append(".\\0\\"+os.path.basename(v))
                        model1.appendRow(QtGui.QStandardItem(".\\0\\"+os.path.basename(v)))
                     elif k == "x'":
                        self.fullFileList2.append(v)
                        model2.appendRow(QtGui.QStandardItem(v))

            self.lv_list1.setModel(model1)            
            self.lv_list2.setModel(model2)  
            self.pbar_doallCal.setMaximum(len(self.fullFileList))

         except Exception as ex:
            print(ex)
            print(sys.exc_info)
            exMsg = QMessageBox.information(self, 'openAll1', str(sys.exc_info()), QMessageBox.Ok )


   def openAllThreshold(self):      
      dirPath=''
      dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(dirPath)
      if dirPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_dirTH.setText(dirPath)
         self.fileListTH = []
         self.fullFileListTH=[]
         self.fileListTH = os.listdir(dirPath)
         self.fileListTH = [file for file in self.fileListTH if file.endswith(".png") or file.endswith(".jpg")]
         self.fileListTH.sort()
         model = QtGui.QStandardItemModel()
         for file in self.fileListTH:
            fullpath = dirPath + "/"+file
            self.fullFileListTH.append (fullpath)
            model.appendRow(QtGui.QStandardItem(file))
         self.lv_listTH.setModel(model)      


   def openAllHisto(self):      
      dirPath=''
      dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(dirPath)
      if dirPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.histoResultName = dirPath.split('/')
         self.histoResultName=self.histoResultName.pop()

         self.te_dirHisto.setText(dirPath)
         self.fileListHisto = []
         self.fullFileListHisto=[]
         self.fileListHisto = os.listdir(dirPath)
         self.fileListHisto = [file for file in self.fileListHisto if file.endswith(".png") or file.endswith(".jpg")]
         self.fileListHisto.sort()
         model = QtGui.QStandardItemModel()
         for file in self.fileListHisto:
            fullpath = dirPath + "/"+file
            self.fullFileListHisto.append (fullpath)
            model.appendRow(QtGui.QStandardItem(file))
         self.lv_listHisto.setModel(model)    


   def openAllBright(self):      
      dirPath=''
      dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(dirPath)
      if dirPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_dirBright.setText(dirPath)
         self.fileListBright = []
         self.fullFileListBright=[]
         self.fileListBright = os.listdir(dirPath)
         self.fileListBright = [file for file in self.fileListBright if file.endswith(".png") or file.endswith(".jpg")]
         self.fileListBright.sort()
         model = QtGui.QStandardItemModel()
         for file in self.fileListBright:
            fullpath = dirPath + "/"+file
            self.fullFileListBright.append (fullpath)
            model.appendRow(QtGui.QStandardItem(file))
         self.lv_listBright.setModel(model)    


   def add(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.add(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def sub(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.subtract(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def mul(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.multiply(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def div(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.divide(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def absdiff(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.absdiff(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def bitand(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.bitwise_and(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def bitor(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.bitwise_or(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def bitnot(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.bitwise_not(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def bitxor(self):
      if self.te_path1_2.toPlainText() !='' and self.te_path2_2.toPlainText() !='' :
         img1 = cv2.imread(self.te_path1_2.toPlainText(),cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(self.te_path2_2.toPlainText(), cv2.IMREAD_GRAYSCALE)
         dst = cv2.bitwise_xor(img1,img2)
         cv2.imshow("Result", dst);
         cv2.waitKey(0)
      else :
         QMessageBox.information(self, 'Result', "Please choose img", QMessageBox.Ok )

   def absdiff2(self,img1path,img2path):
         path = os.path.abspath(self.te_dirCal.toPlainText())
         img1 = cv2.imread(img1path,cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(img2path, cv2.IMREAD_GRAYSCALE)
         #cv2.imshow("img1",img1)
         #cv2.imshow("img2",img2)
         #cv2.waitKey
         dst = cv2.absdiff(img1,img2)
         dst2 = cv2.absdiff(img1,img2)
         filename = str(os.path.splitext(os.path.split(img1path)[1])[0])
         filename2 = str(os.path.splitext(os.path.split(img2path)[1])[0])
         cv2.imwrite(path+'/absdiff/'+filename+'_'+filename2+'_absdiff.png',dst)
         #cv2.imwrite('absdiff2/'+filename+'_'+filename2+'_absdiff2.png',dst2)


   def subtract2(self,img1path,img2path):
         path = os.path.abspath(self.te_dirCal.toPlainText())
         img1 = cv2.imread(img1path,cv2.IMREAD_GRAYSCALE)
         img2 = cv2.imread(img2path, cv2.IMREAD_GRAYSCALE)
         dst = cv2.subtract(img1,img2)
         dst2 = cv2.subtract(img2,img1)
         filename = str(os.path.splitext(os.path.split(img1path)[1])[0])
         filename2 = str(os.path.splitext(os.path.split(img2path)[1])[0])
         cv2.imwrite(path+'/subtract/'+filename+'_'+filename2+'_subtract.png',dst)
         #cv2.imwrite('subtract2/'+filename+'_'+filename2+'_subtract2.png',dst2)
         

   def skimage(self,img1path,img2path):
      path = os.path.abspath(self.te_dirCal.toPlainText())
      grayA = cv2.imread(img1path,cv2.IMREAD_GRAYSCALE)
      grayB = cv2.imread(img2path, cv2.IMREAD_GRAYSCALE)

      (score, diff) = compare_ssim(grayA, grayB, full=True)
      diff = (diff * 255).astype("uint8")
      print("skimageSSIMRate: {}".format(score))

      #thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
      ret,thresh = cv2.threshold(diff, 150, 256, cv2.THRESH_TRUNC)

      filename = str(os.path.splitext(os.path.split(img1path)[1])[0])
      filename2 = str(os.path.splitext(os.path.split(img2path)[1])[0])
      cv2.imwrite(path+'/skimageDiff/'+filename+'_'+filename2+'skimgDiff.png',diff)
      cv2.imwrite(path+'/skimageThreshold/'+filename+'_'+filename2+'skimgThresh.png',thresh)
      return str(score)


   def openDirectory0(self):
      self.folderPath=''
      self.folderPath = (QFileDialog.getOpenFileName(self, "Select file1"))
      print(self.folderPath)
      if self.folderPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_path1.setText(self.folderPath[0])
         self.lb_img1.setPixmap(QtGui.QPixmap(self.folderPath[0]))
         self.lb_img1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_img1.setScaledContents(True)


   def openDirectoryTH(self):
      folderPath=''
      folderPath = (QFileDialog.getOpenFileName(self, "Select file1"))
      print(folderPath)
      if folderPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_pathTH.setText(folderPath[0])
         self.lb_imgTH.setPixmap(QtGui.QPixmap(folderPath[0]))
         self.lb_imgTH.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_imgTH.setScaledContents(True)


   def pattrenMatchSimple(self,img1path,img2path):
      img1 = cv2.imread(img1path,cv2.IMREAD_GRAYSCALE)
      img2 = cv2.imread(img2path, cv2.IMREAD_GRAYSCALE)
      res = cv2.matchTemplate(img1, img2, patternList[self.cb_pattrenMatch.currentIndex()])
      '''
      imgMaster = cv2.imread('0/OK (1).jpg_Num1.png', cv2.IMREAD_GRAYSCALE)
      resMaster= cv2.matchTemplate(imgMaster, img2, patternList[self.cb_pattrenMatch.currentIndex()])
      print(res,resMaster)
      return str(res[0][0]),str(resMaster[0][0])
      '''
      print(res)
      return str(res[0][0])


   def pattrenMatch(self, num,path):
      if num == 1: 
         img1= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
         dst = cv2.imread(path,cv2.IMREAD_COLOR)
      else : 
         img1=cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
         dst = self.dst.copy()

      p1 = cv2.imread('match1.jpg', cv2.IMREAD_GRAYSCALE) #각도 계산할 기준 ROI#1
      p2 = cv2.imread('match2.jpg', cv2.IMREAD_GRAYSCALE) #각도 계산할 기준 ROI#2

      w,h= img1.shape[::-1]
      w1,h1 = p1.shape[::-1]
      w2,h2 = p2.shape[::-1]

      res = cv2.matchTemplate(img1,p1,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left1 = max_loc
      bottom_right1 = (self.top_left1[0]+w1, self.top_left1[1]+h1)

      res = cv2.matchTemplate(img1,p2,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left2 = max_loc
      bottom_right2 = (self.top_left2[0]+w2, self.top_left2[1]+h2)

      # 찾은 템플릿이미지에 흰색 사각형을 쳐줌
      cv2.rectangle(dst, self.top_left1, bottom_right1, 255, 3)
      cv2.rectangle(dst, self.top_left2, bottom_right2, 255, 3)

      cv2.line(dst,self.top_left1,self.top_left2,255,3)
      cv2.line(dst,(0,self.top_left2[1]),(w,self.top_left2[1]),255,3)

      #image = QtGui.QImage(dst.data, w, h, dst.strides[0], QtGui.QImage.Format_Grayscale8)
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=dst


   def findDegree(self):
      def dot(vA, vB):
       return vA[0]*vB[0]+vA[1]*vB[1]

      #vA = [(self.top_left1[0]-self.top_left2[0]), (self.top_left1[1]-self.top_left2[1])]
      vA = [(self.top_left1[0]-self.top_left2[0]), (self.top_left1[1]-self.top_left2[1])]
      vB = [0-(self.top_left2[0]), (self.top_left2[1]-self.top_left2[1])]
      # Get dot prod
      dot_prod = dot(vA, vB)
      # Get magnitudes
      magA = dot(vA, vA)**0.5
      magB = dot(vB, vB)**0.5
      # Get cosine value
      cos_ = dot_prod/magA/magB
      # Get angle in radians and then convert to degrees
      angle = math.acos(dot_prod/magB/magA)
      # Basically doing angle <- angle mod 360
      ang_deg = math.degrees(angle)%360

      if ang_deg-180>=0:
         # As in if statement
         print( "over 180")
         #print( 360 - ang_deg)
         self.te_degree.setText(str(360 - ang_deg))
      else: 
         #print( ang_deg)
         self.te_degree.setText(str(ang_deg))

      self.te_mvDegree.setText(str(49.07594497393691-ang_deg))
      print(self.te_mvDegree.toPlainText())


   def rotate(self,num,path):
      if num == 1: 
         src = cv2.imread(path,cv2.IMREAD_COLOR)
      else : 
         src = self.dst.copy()

      height, width, channel = src.shape
      matrix = cv2.getRotationMatrix2D((width/2, height/2),  360-float(self.te_mvDegree.toPlainText()), 1)
      self.dst = cv2.warpAffine(src, matrix, (width, height))
      '''
      cv2.imshow("src", src)
      cv2.imshow("dst", self.dst)
      cv2.waitKey(0)
      '''
      image = QtGui.QImage(self.dst.data,self.dst.shape[1], self.dst.shape[0], self.dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=self.dst


   def roiSet(self):
      img1= cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
      dst2 = self.dst.copy()

      p1 = cv2.imread('match1.jpg', cv2.IMREAD_GRAYSCALE)
      p2 = cv2.imread('match2.jpg', cv2.IMREAD_GRAYSCALE)

      w,h= img1.shape[::-1]
      w1,h1 = p1.shape[::-1]
      w2,h2 = p2.shape[::-1]

      res = cv2.matchTemplate(img1,p1,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left1 = max_loc
      bottom_right1 = (self.top_left1[0]+w1, self.top_left1[1]+h1)

      res = cv2.matchTemplate(img1,p2,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left2 = max_loc
      bottom_right2 = (self.top_left2[0]+w2, self.top_left2[1]+h2)

      cv2.rectangle(dst2, self.top_left1, bottom_right1, 255, 3)
      cv2.rectangle(dst2, self.top_left2, bottom_right2, 255, 3)

      cv2.line(dst2,self.top_left1,self.top_left2,255,3)
      cv2.line(dst2,(0,self.top_left2[1]),(w,self.top_left2[1]),255,3)


      self.x1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.y1 = self.top_left1[1]+int(self.te_startY.toPlainText())
      self.x2 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.y2 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.lb_startX.setText(str(self.x1))
      self.lb_startY.setText(str(self.y1))
      self.lb_endX.setText(str(self.x2))
      self.lb_endY.setText(str(self.y2))

      cv2.rectangle(dst2, (self.x1,self.y1),(self.x2,self.y2), (255,255,255), 3)
      image = QtGui.QImage(dst2.data,dst2.shape[1], dst2.shape[0], dst2.strides[0], QtGui.QImage.Format_RGB888)

      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=dst2


   def roiCut(self):
      self.dstRst=self.dst.copy()
      self.dstRst = self.dst[self.y1:self.y2,self.x1:self.x2]
      #cv2.imshow("test",self.dstRst)
      dst = self.dstRst.copy()
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=self.dstRst


   def save(self,path,num):
      if num == 1:
         if (path) != '':
            filename=os.path.splitext(os.path.basename(path))[0]
            root = Tk().withdraw()
            title = 'Save project as'
            ftypes = [('jpg file', '.jpg')]
            filePath = tkinter.filedialog.asksaveasfilename(filetypes=ftypes, title=title,
                                                                    initialfile=filename)
            if filePath != '':
               scipy.misc.imsave(filePath+"_new.jpg", self.dstRst)
               QMessageBox.information(self, 'save', 'Save Complete', QMessageBox.Ok )
      else :
         scipy.misc.imsave(path+"_new.jpg", self.dstRst)
         print(path+"_new.jpg")


   def roiSet2(self):
      img1= cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
      dst2 = self.dst.copy()

      p1 = cv2.imread('match1.jpg', cv2.IMREAD_GRAYSCALE)
      p2 = cv2.imread('match2.jpg', cv2.IMREAD_GRAYSCALE)

      w,h= img1.shape[::-1]
      w1,h1 = p1.shape[::-1]
      w2,h2 = p2.shape[::-1]

      res = cv2.matchTemplate(img1,p1,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left1 = max_loc
      bottom_right1 = (self.top_left1[0]+w1, self.top_left1[1]+h1)

      res = cv2.matchTemplate(img1,p2,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left2 = max_loc
      bottom_right2 = (self.top_left2[0]+w2, self.top_left2[1]+h2)

      cv2.rectangle(dst2, self.top_left1, bottom_right1, 255, 3)
      cv2.rectangle(dst2, self.top_left2, bottom_right2, 255, 3)

      cv2.line(dst2,self.top_left1,self.top_left2,255,3)
      cv2.line(dst2,(0,self.top_left2[1]),(w,self.top_left2[1]),255,3)


      self.x1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.y1 = self.top_left1[1]+int(self.te_startY.toPlainText())
      self.x2 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.y2 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.lb_startX.setText(str(self.x1))
      self.lb_startY.setText(str(self.y1))
      self.lb_endX.setText(str(self.x2))
      self.lb_endY.setText(str(self.y2))

      self.x1_half = int(((self.x2-self.x1)/2)+self.x1)
      self.y1_half = int(((self.y2-self.y1)/2)+self.y1)


      cv2.rectangle(dst2, (self.x1,self.y1),(self.x1_half,self.y1_half), (255,0,255), 2) #좌상
      cv2.rectangle(dst2, (self.x1_half,self.y1),(self.x2,self.y1_half), (255,80,255), 2) #우상
      cv2.rectangle(dst2, (self.x1,self.y1_half),(self.x1_half,self.y2), (255,160,255), 2) #좌하
      cv2.rectangle(dst2, (self.x1_half,self.y1_half),(self.x2,self.y2), (255,40,255), 2) #우하
      image = QtGui.QImage(dst2.data,dst2.shape[1], dst2.shape[0], dst2.strides[0], QtGui.QImage.Format_RGB888)

      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=dst2


   def roiCut2(self,path,num):
      self.dstRst1=self.dst.copy()
      self.dstRst2=self.dst.copy()
      self.dstRst3=self.dst.copy()
      self.dstRst4=self.dst.copy()

      self.dstRst1 = self.dst[self.y1:self.y1_half,self.x1:self.x1_half]#좌상
      self.dstRst2 = self.dst[self.y1:self.y1_half,self.x1_half:self.x2]#우상
      self.dstRst3 = self.dst[self.y1_half:self.y2,self.x1:self.x1_half]#좌하
      self.dstRst4 = self.dst[self.y1_half:self.y2,self.x1_half:self.x2]#우하

      if num == 0:
         scipy.misc.imsave("LEFT_UP.jpg", self.dstRst1)
         scipy.misc.imsave("RIGHT_UP.jpg", self.dstRst2)
         scipy.misc.imsave("LEFT_DOWN.jpg", self.dstRst3)
         scipy.misc.imsave("RIGHT_DOWN.jpg", self.dstRst4)
      elif num ==1:
         scipy.misc.imsave(path+"_LEFT_UP.jpg", self.dstRst1)
         print(path+"_LEFT_UP.jpg")
         scipy.misc.imsave(path+"_RIGHT_UP.jpg", self.dstRst2)
         print(path+"_RIGHT_UP.jpg")
         scipy.misc.imsave(path+"_LEFT_DOWN.jpg", self.dstRst3)
         print(path+"_LEFT_DOWN.jpg")
         scipy.misc.imsave(path+"_RIGHT_DOWN.jpg", self.dstRst4)
         print(path+"_RIGHT_DOWN.jpg")

      '''
      cv2.imshow("LEFT_UP",self.dstRst1)
      cv2.imshow("RIGHT_UP",self.dstRst2)   
      cv2.imshow("LEFT_DOWN",self.dstRst3)   
      cv2.imshow("RIGHT_DOWN",self.dstRst4)    

      dst = self.dstRst.copy()
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=self.dstRst
      '''

   def roiSet3(self):
      img1= cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
      dst2 = self.dst.copy()

      p1 = cv2.imread('match1.jpg', cv2.IMREAD_GRAYSCALE)
      p2 = cv2.imread('match2.jpg', cv2.IMREAD_GRAYSCALE)

      w,h= img1.shape[::-1]
      w1,h1 = p1.shape[::-1]
      w2,h2 = p2.shape[::-1]

      res = cv2.matchTemplate(img1,p1,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left1 = max_loc
      bottom_right1 = (self.top_left1[0]+w1, self.top_left1[1]+h1)

      res = cv2.matchTemplate(img1,p2,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left2 = max_loc
      bottom_right2 = (self.top_left2[0]+w2, self.top_left2[1]+h2)

      cv2.rectangle(dst2, self.top_left1, bottom_right1, 255, 3)
      cv2.rectangle(dst2, self.top_left2, bottom_right2, 255, 3)

      cv2.line(dst2,self.top_left1,self.top_left2,255,3)
      cv2.line(dst2,(0,self.top_left2[1]),(w,self.top_left2[1]),255,3)

      
      self.x1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.y1 = self.top_left1[1]+int(self.te_startY.toPlainText())
      self.x2 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.y2 = self.top_left2[1]+int(self.te_endY.toPlainText())


      self.xx1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.yy1 = self.top_left1[1]+int(self.te_startY.toPlainText())

      self.xx2 = int(((self.x2-self.x1)/3)+self.x1)
      self.yy2 = int(((self.y2-self.y1)/3)+self.y1)

      self.xx3 = int(((self.x2-self.x1)/3)*2+self.x1)
      self.yy3 = int(((self.y2-self.y1)/3)*2+self.y1)

      self.xx4 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.yy4 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.lb_startX.setText(str(self.xx1))
      self.lb_startY.setText(str(self.yy1))
      self.lb_endX.setText(str(self.xx4))
      self.lb_endY.setText(str(self.yy4))


      cv2.rectangle(dst2, (self.xx1,self.yy1),(self.xx2,self.yy2), (255,0,255), 2) #1
      cv2.rectangle(dst2, (self.xx2,self.yy1),(self.xx3,self.yy2), (255,80,255), 2) #2
      cv2.rectangle(dst2, (self.xx3,self.yy1),(self.xx4,self.yy2), (255,80,255), 2) #3
      cv2.rectangle(dst2, (self.xx1,self.yy2),(self.xx2,self.yy3), (255,0,255), 2) #4
      cv2.rectangle(dst2, (self.xx2,self.yy2),(self.xx3,self.yy3), (255,80,255), 2) #5
      cv2.rectangle(dst2, (self.xx3,self.yy2),(self.xx4,self.yy3), (255,80,255), 2) #6
      cv2.rectangle(dst2, (self.xx1,self.yy3),(self.xx2,self.yy4), (255,0,255), 2) #7
      cv2.rectangle(dst2, (self.xx2,self.yy3),(self.xx3,self.yy4), (255,80,255), 2) #8
      cv2.rectangle(dst2, (self.xx3,self.yy3),(self.xx4,self.yy4), (255,80,255), 2) #9


      #cv2.rectangle(dst2, (self.x1,self.y1_half),(self.x2_half,self.y2), (255,160,255), 2) #좌하
      #cv2.rectangle(dst2, (self.x1_half,self.y1_half),(self.x2,self.y2), (255,40,255), 2) #우하


      image = QtGui.QImage(dst2.data,dst2.shape[1], dst2.shape[0], dst2.strides[0], QtGui.QImage.Format_RGB888)

      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=dst2


   def roiSet4(self,path):
      self.dst = cv2.imread(path,cv2.IMREAD_COLOR)
      img1= cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
      dst2 = self.dst.copy()

      p1 = cv2.imread('match1.jpg', cv2.IMREAD_GRAYSCALE)
      p2 = cv2.imread('match2.jpg', cv2.IMREAD_GRAYSCALE)

      w,h= img1.shape[::-1]
      w1,h1 = p1.shape[::-1]
      w2,h2 = p2.shape[::-1]

      res = cv2.matchTemplate(img1,p1,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left1 = max_loc
      bottom_right1 = (self.top_left1[0]+w1, self.top_left1[1]+h1)

      res = cv2.matchTemplate(img1,p2,cv2.TM_CCORR_NORMED)
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      self.top_left2 = max_loc
      bottom_right2 = (self.top_left2[0]+w2, self.top_left2[1]+h2)

      cv2.rectangle(dst2, self.top_left1, bottom_right1, 255, 3)
      cv2.rectangle(dst2, self.top_left2, bottom_right2, 255, 3)

      cv2.line(dst2,self.top_left1,self.top_left2,255,3)
      cv2.line(dst2,(0,self.top_left2[1]),(w,self.top_left2[1]),255,3)

      self.x1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.y1 = self.top_left1[1]+int(self.te_startY.toPlainText())

      self.x2 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.y2 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.xx1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.yy1 = self.top_left1[1]+int(self.te_startY.toPlainText())

      self.xx2 = int(((self.x2-self.x1)/3)+self.x1)
      self.yy2 = int(((self.y2-self.y1)/3)+self.y1)

      self.xx3 = int(((self.x2-self.x1)/3)*2+self.x1)
      self.yy3 = int(((self.y2-self.y1)/3)*2+self.y1)

      self.xx4 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.yy4 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.lb_startX.setText(str(self.xx1))
      self.lb_startY.setText(str(self.yy1))
      self.lb_endX.setText(str(self.xx4))
      self.lb_endY.setText(str(self.yy4))

      cv2.rectangle(dst2, (self.xx1,self.yy1),(self.xx2,self.yy2), (255,0,255), 2) #1
      cv2.rectangle(dst2, (self.xx2,self.yy1),(self.xx3,self.yy2), (255,80,255), 2) #2
      cv2.rectangle(dst2, (self.xx3,self.yy1),(self.xx4,self.yy2), (255,80,255), 2) #3
      cv2.rectangle(dst2, (self.xx1,self.yy2),(self.xx2,self.yy3), (255,0,255), 2) #4
      cv2.rectangle(dst2, (self.xx2,self.yy2),(self.xx3,self.yy3), (255,80,255), 2) #5
      cv2.rectangle(dst2, (self.xx3,self.yy2),(self.xx4,self.yy3), (255,80,255), 2) #6
      cv2.rectangle(dst2, (self.xx1,self.yy3),(self.xx2,self.yy4), (255,0,255), 2) #7
      cv2.rectangle(dst2, (self.xx2,self.yy3),(self.xx3,self.yy4), (255,80,255), 2) #8
      cv2.rectangle(dst2, (self.xx3,self.yy3),(self.xx4,self.yy4), (255,80,255), 2) #9

      #cv2.rectangle(dst2, (self.x1,self.y1_half),(self.x2_half,self.y2), (255,160,255), 2) #좌하
      #cv2.rectangle(dst2, (self.x1_half,self.y1_half),(self.x2,self.y2), (255,40,255), 2) #우하

      image = QtGui.QImage(dst2.data,dst2.shape[1], dst2.shape[0], dst2.strides[0], QtGui.QImage.Format_RGB888)

      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=dst2


   def roiCut3(self,path,num):
      self.dstRst1=self.dst.copy()
      self.dstRst2=self.dst.copy()
      self.dstRst3=self.dst.copy()
      self.dstRst4=self.dst.copy()
      '''
      cv2.rectangle(dst2, (self.xx1,self.yy1),(self.xx2,self.yy2), (255,0,255), 2) #1
      cv2.rectangle(dst2, (self.xx2,self.yy1),(self.xx3,self.yy2), (255,80,255), 2) #2
      cv2.rectangle(dst2, (self.xx3,self.yy1),(self.xx4,self.yy2), (255,80,255), 2) #3
      cv2.rectangle(dst2, (self.xx1,self.yy2),(self.xx2,self.yy3), (255,0,255), 2) #4
      cv2.rectangle(dst2, (self.xx2,self.yy2),(self.xx3,self.yy3), (255,80,255), 2) #5
      cv2.rectangle(dst2, (self.xx3,self.yy2),(self.xx4,self.yy3), (255,80,255), 2) #6
      cv2.rectangle(dst2, (self.xx1,self.yy3),(self.xx2,self.yy4), (255,0,255), 2) #7
      cv2.rectangle(dst2, (self.xx2,self.yy3),(self.xx3,self.yy4), (255,80,255), 2) #8
      cv2.rectangle(dst2, (self.xx3,self.yy3),(self.xx4,self.yy4), (255,80,255), 2) #9
      '''
      self.dstRst1 = self.dst[self.yy1:self.yy2,self.xx1:self.xx2]#1
      self.dstRst2 = self.dst[self.yy1:self.yy2,self.xx2:self.xx3]#2
      self.dstRst3 = self.dst[self.yy1:self.yy2,self.xx3:self.xx4]#3
      self.dstRst4 = self.dst[self.yy2:self.yy3,self.xx1:self.xx2]#4
      self.dstRst5 = self.dst[self.yy2:self.yy3,self.xx2:self.xx3]#5
      self.dstRst6 = self.dst[self.yy2:self.yy3,self.xx3:self.xx4]#6
      self.dstRst7 = self.dst[self.yy3:self.yy4,self.xx1:self.xx2]#7
      self.dstRst8 = self.dst[self.yy3:self.yy4,self.xx2:self.xx3]#8
      self.dstRst9 = self.dst[self.yy3:self.yy4,self.xx3:self.xx4]#9


      if num == 0:
         scipy.misc.imsave("Num1.jpg", self.dstRst1)
         scipy.misc.imsave("Num2.jpg", self.dstRst2)
         scipy.misc.imsave("Num3.jpg", self.dstRst3)
         scipy.misc.imsave("Num4.jpg", self.dstRst4)
         scipy.misc.imsave("Num5.jpg", self.dstRst5)
         scipy.misc.imsave("Num6.jpg", self.dstRst6)
         scipy.misc.imsave("Num7.jpg", self.dstRst7)
         scipy.misc.imsave("Num8.jpg", self.dstRst8)
         scipy.misc.imsave("Num9.jpg", self.dstRst9)
      elif num ==1:
         scipy.misc.imsave(path+"_Num1.jpg", self.dstRst1)
         print(path+"_Num1.jpg")
         scipy.misc.imsave(path+"_Num2.jpg", self.dstRst2)
         print(path+"_Num2.jpg")
         scipy.misc.imsave(path+"_Num3.jpg", self.dstRst3)
         print(path+"_Num3.jpg")
         scipy.misc.imsave(path+"_Num4.jpg", self.dstRst4)
         print(path+"_Num4.jpg")
         scipy.misc.imsave(path+"_Num5.jpg", self.dstRst5)
         print(path+"_Num5.jpg")
         scipy.misc.imsave(path+"_Num6.jpg", self.dstRst6)
         print(path+"_Num6.jpg")
         scipy.misc.imsave(path+"_Num7.jpg", self.dstRst7)
         print(path+"_Num7.jpg")
         scipy.misc.imsave(path+"_Num8.jpg", self.dstRst8)
         print(path+"_Num8.jpg")
         scipy.misc.imsave(path+"_Num9.jpg", self.dstRst9)
         print(path+"_Num9.jpg")
      '''
      cv2.imshow("LEFT_UP",self.dstRst1)
      cv2.imshow("RIGHT_UP",self.dstRst2)   
      cv2.imshow("LEFT_DOWN",self.dstRst3)   
      cv2.imshow("RIGHT_DOWN",self.dstRst4)    

      dst = self.dstRst.copy()
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)
      self.dstBig=self.dstRst
      '''


   def roiload(self,mode):       
         try:
            self.jsonPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings','settings.json')
            if mode == 0:
               None
            elif mode == 1:
               self.folderPath=''
               self.folderPath = (QFileDialog.getOpenFileName(self, "Select file1"))
               print(self.folderPath)
               if self.folderPath is '':
                  buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose jsonFile", QMessageBox.Ok )
               else:
                  self.jsonPath = self.folderPath[0]

            self.te_roiPath.setText(self.jsonPath)
            with open(self.jsonPath, 'r') as f:
               self.jsonData = json.load(f, object_pairs_hook=OrderedDict)

            self.te_startX.setText(str(self.jsonData['Option']['startX']))
            self.te_startY.setText(str(self.jsonData['Option']['startY']))
            self.te_endX.setText(str(self.jsonData['Option']['endX']))
            self.te_endY.setText(str(self.jsonData['Option']['endY']))

         except Exception as ex:
            self.jsonDataMain=OrderedDict()
            self.jsonData=OrderedDict()
            self.jsonData['startX']=0
            self.jsonData['startY']=0
            self.jsonData['endX']=0
            self.jsonData['endY']=0
            self.jsonDataMain["Option"]=self.jsonData
            with open(self.jsonPath, 'w') as make_file:
               json.dump(self.jsonDataMain, make_file,ensure_ascii=False,indent=4)
            with open(self.jsonPath, 'r') as f:
               self.jsonData = json.load(f, object_pairs_hook=OrderedDict)
           

   def roiSave(self):
         print(self.jsonData.keys())
         self.jsonData['Option']['startX']=self.te_startX.toPlainText()
         self.jsonData['Option']['startY']=self.te_startY.toPlainText()
         self.jsonData['Option']['endX']=self.te_endX.toPlainText()
         self.jsonData['Option']['endY']=self.te_endY.toPlainText()
         with open(self.jsonPath, 'w') as outfile:
            json.dump(self.jsonData, outfile,indent=4)
         QMessageBox.information(self, 'PyQt5 message', self.jsonPath+' Save Complete', QMessageBox.Ok )


   def openDirectoryALL(self):
      self.dirPath=''
      self.dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(self.dirPath)
      if self.dirPath is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_path3.setText(self.dirPath)
         self.fileList =   []
         self.fullFileList=[]
         self.fileList = os.listdir(self.dirPath)
         self.fileList.sort()
         model = QtGui.QStandardItemModel()
         for file in self.fileList:
            fullpath = self.dirPath + "/"+file
            self.fullFileList.append (fullpath)
            model.appendRow(QtGui.QStandardItem(file))
         self.listView.setModel(model)


   def doAll(self):
      for file in self.fullFileList:
         self.te_path1.setText(file)
         self.lb_img1.setPixmap(QtGui.QPixmap(file))
         self.lb_img1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_img1.setScaledContents(True)

         cnt=1
         print(cnt)
         self.pattrenMatch(num=1,path=file)
         time.sleep(0.1)
         self.findDegree()
         time.sleep(0.1)
         self.rotate(num=1,path=file)
         time.sleep(0.1)

         while cnt<3:
            cnt=cnt+1
            print(cnt)
            self.pattrenMatch(num=2,path=None)
            time.sleep(0.1)
            self.findDegree()
            time.sleep(0.1)
            self.rotate(num=2,path=None)
            time.sleep(0.1)


         '''
         oldDegree = None
         while abs(float(self.te_mvDegree.toPlainText())) != 0.0 :
         #while abs(float(self.te_mvDegree.toPlainText())) >= 0.005 :
            nowDegree = abs(float(self.te_mvDegree.toPlainText()))
            print( oldDegree,nowDegree)
            if nowDegree == oldDegree and nowDegree < 0.01:
               break
            cnt=cnt+1
            print(cnt)
            self.pattrenMatch(num=2,path=None)
            time.sleep(0.1)
            oldDegree = abs(float(self.te_mvDegree.toPlainText()))
            self.findDegree()
            time.sleep(0.1)
            self.rotate(num=2,path=None)
            time.sleep(0.1)
         '''

         print("")
         '''
         while float(self.te_mvDegree.toPlainText()) != 0.0 :
            cnt=cnt+1
            if cnt>50 and abs(float(self.te_mvDegree.toPlainText())) < 0.005:
               break
            print(cnt)
            self.pattrenMatch(num=2,path=None)
            time.sleep(0.1)
            self.findDegree()
            time.sleep(0.1)
            self.rotate(num=2,path=None)
            time.sleep(0.1)
         '''

         self.roiSet()
         time.sleep(0.1)
         self.roiCut()
         time.sleep(0.1)
         self.save(num=2,path=file)

      print("doAll work Finsh")
      QMessageBox.information(self, 'doAll', "work Finsh", QMessageBox.Ok )

   def doAllCal(self):
         print("Start")
         self.te_log.setText("Start")
         self.te_log.ensureCursorVisible()
         self.te_log.setTextCursor(self.te_log.textCursor())
         self.pbar_doallCal.setValue(0)
         path = os.path.abspath(self.te_dirCal.toPlainText())
         try:
             if os.path.isfile(path+'/result.csv'):
                os.remove(path+'/result.csv')
                time.sleep(0.05)

             if os.path.isdir(path+'/subtract'):
                shutil.rmtree(os.path.join(path+'/subtract'))
                time.sleep(0.05)
                os.makedirs(os.path.join(path+'/subtract'))
                time.sleep(0.05)
             elif not(os.path.isdir(path+'/subtract')):
                 os.makedirs(os.path.join(path+'/subtract'))
                 time.sleep(0.05)

             if os.path.isdir(path+'/absdiff'):
                 shutil.rmtree(os.path.join(path+'/absdiff'))
                 time.sleep(0.05)
                 os.makedirs(os.path.join(path+'/absdiff'))
                 time.sleep(0.05)
             elif not(os.path.isdir(path+'/absdiff')):
                 os.makedirs(os.path.join(path+'/absdiff'))
                 time.sleep(0.05)

             if os.path.isdir(path+'/skimageDiff'):
                shutil.rmtree(os.path.join(path+'/skimageDiff'))
                time.sleep(0.05)
                os.makedirs(os.path.join(path+'/skimageDiff'))
                time.sleep(0.05)
             elif not(os.path.isdir(path+'/skimageDiff')):
                 os.makedirs(os.path.join(path+'/skimageDiff'))
                 time.sleep(0.05)

             if os.path.isdir(path+'/skimageThreshold'):
                shutil.rmtree(os.path.join(path+'/skimageThreshold'))
                time.sleep(0.05)
                os.makedirs(os.path.join(path+'/skimageThreshold'))
                time.sleep(0.05)
             elif not(os.path.isdir(path+'/skimageThreshold')):
                 os.makedirs(os.path.join(path+'/skimageThreshold'))
                 time.sleep(0.05)

         except OSError as e:
             if e.errno != errno.EEXIST:
                 print("Failed to create directory!!!!!")
                 raise


         try:
            lenght=len(self.fullFileList)
            rst= self.cb_pattrenMatch.currentText()+"\nx,x1,openCvTemplateMatch,skimageSSIM,tensorSSIM\n"
            for a in range(lenght):
               img1=self.fullFileList.pop()
               img2=self.fullFileList2.pop()
               print (img1[2:],img2[2:])
               self.te_log.setText(self.te_log.toPlainText()+"\n--------------------------")
               self.te_log.setText(self.te_log.toPlainText()+"\n"+"["+img1[2:]+"]"+","+"["+img2[2:]+"]")
               self.te_log.moveCursor(QtGui.QTextCursor.End)
               img1=os.path.abspath(path+"/"+img1[2:])
               img2=os.path.abspath(path+"/"+img2[2:])
               if os.path.isfile(img1) == False:
                  print("img1 isfile false")
                  self.te_log.setText(self.te_log.toPlainText()+"\n"+"img1 isfile false")
                  break
               if os.path.isfile(img2) == False:
                  print("img2 isfile false")
                  self.te_log.setText(self.te_log.toPlainText()+"\n"+"img2 isfile false")
                  break
               self.absdiff2(img1,img2)
               time.sleep(0.05)
               self.subtract2(img1,img2)
               time.sleep(0.05)
               matchrst = self.pattrenMatchSimple(img1,img2)   
               self.te_log.setText(self.te_log.toPlainText()+"\n"+"TemplateMatchRate:"+matchrst)
               self.te_log.moveCursor(QtGui.QTextCursor.End)
               time.sleep(0.05)
               skirst = self.skimage(img1,img2)
               self.te_log.setText(self.te_log.toPlainText()+"\n"+"SkimageSSIMRate:"+skirst)
               self.te_log.moveCursor(QtGui.QTextCursor.End)
               time.sleep(0.05)
               tensorrst="0"

               tensorrst = self.tensorSSIM(img1,img2)
               self.te_log.setText(self.te_log.toPlainText()+"\n"+"TensorSSIMRate:"+tensorrst)
               self.te_log.setText(self.te_log.toPlainText()+"\n--------------------------")
               self.te_log.moveCursor(QtGui.QTextCursor.End)
               time.sleep(0.05)

               rst = rst+img1+","+img2+","+matchrst+","+skirst+","+tensorrst+"\n"
               with open(path+"/result.csv", mode="w") as file:
                  file.writelines(rst)       

               self.pbar_doallCal.setValue(a)
               QMainWindow.update(self)
               QApplication.processEvents()

            emtpymodel = QtGui.QStandardItemModel()
            self.lv_list1.setModel(emtpymodel)            
            self.lv_list2.setModel(emtpymodel)  
            self.pbar_doallCal.setValue(lenght)
            print("Complete!")
            self.te_log.setText(self.te_log.toPlainText()+"\nComplete")
            self.te_log.moveCursor(QtGui.QTextCursor.End)
            QMessageBox.information(self, 'doall', "Complete!!", QMessageBox.Ok )

         except Exception as ex:
            print(ex)
            print(sys.exc_info)
            exMsg = QMessageBox.information(self, 'doAllCal', str(sys.exc_info()), QMessageBox.Ok )

   def addToDoList(self,function):
      print(function)
      if function is '':
         None
         #buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         try:
            self.filterDoList.append(function)
            self.filterDoListModel.appendRow(QtGui.QStandardItem(function))
            self.lv_listTH_2.setModel(self.filterDoListModel)

         except Exception as ex:
            print(ex)
            print(sys.exc_info)
            exMsg = QMessageBox.information(self, 'addToDoList', str(sys.exc_info()), QMessageBox.Ok )

   def doAllThreshold(self):
      print("start")
      if len(self.filterDoList) == 0:
          None
      else:
          try:
             if not(os.path.isdir(self.te_dirTH.toPlainText()+'/filter')):
                os.makedirs(os.path.join(self.te_dirTH.toPlainText()+'/filter'))
          except OSError as e:
             if e.errno != errno.EEXIST:
                print("Failed to create directory!!!!!")
                raise
          try:
             for file in self.fullFileListTH:
                print(file)
                img = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
                itemlist=""
                for item in self.filterDoList:
                    itemlist=itemlist+item
                    if item == "Threshold":
                        img = self.threshold2(img)
                    elif item == "Filter":
                        img = self.filter2(img)
                    elif item == "Open":
                        img = self.morpOpen2(img)
                    elif item == "Close":
                        img = self.morpClose2(img)
                    elif item == "Erosion":
                        img = self.morpErosion2(img)
                    elif item == "Dilation":
                        img = self.morpDilation2(img)
                    elif item == "Mask":
                        img = self.mask2(img)
                    elif item == "GRADIENT":
                        img = self.morpGRADIENT2(img)
                    elif item == "ColorReverse":
                        img = self.colorReverse2(img)
                    elif item == "EqualizeHist":
                        img = self.equalizeHist2(img)
                    elif item == "Sharpening":
                        img = self.sharpening2(img)
                    time.sleep(0.1)

                filename = str(os.path.splitext(os.path.split(file)[1])[0])+item
                cv2.imwrite(os.path.join(self.te_dirTH.toPlainText())+'/filter/'+filename+'_(+'+itemlist+').png',img)
                print('filter/'+filename+'_(+'+itemlist+').png')    
                time.sleep(0.1)

             self.resetDoList()
             emtpymodel = QtGui.QStandardItemModel()
             self.lv_listTH.setModel(emtpymodel)   
             print("Complete!")
             QMessageBox.information(self, 'doall', "Complete!!", QMessageBox.Ok )

          except Exception as ex:
             print(ex)
             print(sys.exc_info)
             exMsg = QMessageBox.information(self, 'doAllTH', str(sys.exc_info()), QMessageBox.Ok )


   def doAllHisto(self,value1,value2):
      print("start")
      rst= "img,histValue\n"  
      path = os.path.abspath(os.path.join(self.te_dirHisto.toPlainText(), os.pardir))

      for a in range(len(self.fullFileListHisto)):
         file=self.fullFileListHisto.pop()
         print(file)
         rst= rst+file+","+self.histo(file,value1,value2)+"\n"
      
      with open(path+"/"+self.histoResultName+"_histogram.csv", mode="w") as file:
         file.writelines(rst)
         
      print("Complete!")
      QMessageBox.information(self, 'doAllHisto', "Complete!!", QMessageBox.Ok )
      time.sleep(0.1)
      self.histoResultName = ''
      self.te_dirHisto.setText('')
      emtpymodel = QtGui.QStandardItemModel()
      self.lv_listHisto.setModel(emtpymodel)   


   def doAllBright(self):
      print("start")
      path = os.path.abspath(os.path.join(self.te_dirBright.toPlainText(), os.pardir))

      if os.path.isdir(path+'/brightChange'):
         shutil.rmtree(os.path.join(path+'/brightChange'))
         time.sleep(0.1)
         os.makedirs(os.path.join(path+'/brightChange'))
      elif not(os.path.isdir(path+'/brightChange')):
         os.makedirs(os.path.join(path+'/brightChange'))

      for a in range(len(self.fullFileListBright)):
         file=self.fullFileListBright.pop()
         if self.histNum1Bright.value() != self.histNum2Bright.value():
            var = random.randrange(self.histNum1Bright.value(),self.histNum2Bright.value())
         elif self.histNum1Bright.value() == self.histNum2Bright.value():
            var = self.histNum1Bright.value()
         self.brightChange(file,var)
         time.sleep(0.1)

      print("Complete!")
      QMessageBox.information(self, 'doAllBright', "Complete!!", QMessageBox.Ok )
      time.sleep(0.1)
      self.te_dirBright.setText('')
      emtpymodel = QtGui.QStandardItemModel()
      self.lv_listBright.setModel(emtpymodel)   


   def doAllScale(self):
      print("start")
      path = os.path.abspath(os.path.join(self.te_dirBright.toPlainText(), os.pardir))

      if os.path.isdir(path+'/ScaleChange'):
         shutil.rmtree(os.path.join(path+'/ScaleChange'))
         time.sleep(0.1)
         os.makedirs(os.path.join(path+'/ScaleChange'))
      elif not(os.path.isdir(path+'/ScaleChange')):
         os.makedirs(os.path.join(path+'/ScaleChange'))

      for a in range(len(self.fullFileListBright)):
         file=self.fullFileListBright.pop()
         if self.ScaleNum1.value() != self.ScaleNum2.value():
            var = random.randrange(self.ScaleNum1.value(),self.ScaleNum2.value())
         elif self.ScaleNum1.value() == self.ScaleNum2.value():
            var = self.ScaleNum1.value()
         self.scaleChange(file,round(var*0.01,2))
         time.sleep(0.1)

      print("Complete!")
      QMessageBox.information(self, 'doAllScale', "Complete!!", QMessageBox.Ok )
      time.sleep(0.1)
      self.te_dirBright.setText('')
      emtpymodel = QtGui.QStandardItemModel()
      self.lv_listBright.setModel(emtpymodel)   


   def doAll2(self):
      for file in self.fullFileList:
         self.te_path1.setText(file)
         self.lb_img1.setPixmap(QtGui.QPixmap(file))
         self.lb_img1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_img1.setScaledContents(True)

         cnt=1
         print(cnt)
         self.pattrenMatch(num=1,path=file)
         time.sleep(0.1)
         self.findDegree()
         time.sleep(0.1)
         self.rotate(num=1,path=file)
         time.sleep(0.1)

         while cnt<3:
            cnt=cnt+1
            print(cnt)
            self.pattrenMatch(num=2,path=None)
            time.sleep(0.1)
            self.findDegree()
            time.sleep(0.1)
            self.rotate(num=2,path=None)
            time.sleep(0.1)

         print("")

         self.roiSet2()
         time.sleep(0.1)
         self.roiCut2(num=1,path=file)

      print("doAll 1/4 work Finsh")
      QMessageBox.information(self, 'doAll', "work Finsh", QMessageBox.Ok )


   def doAll3(self):
      for file in self.fullFileList:
         self.te_path1.setText(file)
         self.lb_img1.setPixmap(QtGui.QPixmap(file))
         self.lb_img1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_img1.setScaledContents(True)

         cnt=1
         print(cnt)
         self.pattrenMatch(num=1,path=file)
         time.sleep(0.1)
         self.findDegree()
         time.sleep(0.1)
         self.rotate(num=1,path=file)
         time.sleep(0.1)

         while cnt<3:
            cnt=cnt+1
            print(cnt)
            self.pattrenMatch(num=2,path=None)
            time.sleep(0.1)
            self.findDegree()
            time.sleep(0.1)
            self.rotate(num=2,path=None)
            time.sleep(0.1)

         print("")

         self.roiSet3()
         time.sleep(0.1)
         self.roiCut3(num=1,path=file)

      print("doAll 1/9 work Finsh")
      QMessageBox.information(self, 'doAll', "work Finsh", QMessageBox.Ok )


   def doOnlyCrop9(self):
      for file in self.fullFileList:
         self.roiSet4(file)
         time.sleep(0.1)
         self.roiCut3(num=1,path=file)

      print("doAll 1/9 work Finsh")
      QMessageBox.information(self, 'doAll', "work Finsh", QMessageBox.Ok )


   def openMainDir(self):
      os.startfile((os.path.dirname(os.path.abspath(__file__)))) 


   def histo(self,path,value1,value2):
       img = cv2.imread(path ,cv2.IMREAD_GRAYSCALE)     
       hist1 = cv2.calcHist([img],[0],None,[256],[0,256])      
       sum=0
       for x in range(value1,value2):
         sum=sum+hist1[x]
       return str(sum[0])

   def histogramTest(self):
       img = cv2.imread(self.te_pathHistoTest.toPlainText() ,cv2.IMREAD_GRAYSCALE)     
       hist1 = cv2.calcHist([img],[0],None,[256],[0,256])      

       self.histoWidget.axes.cla()
       self.histoWidget.axes.plot(hist1,color ="g" )
       #self.histoWidget.axes.hist(img,range=(0,256),bins=256 )
       self.histoWidget.axes.text(0, 1.03, "Histogram", fontsize=13, fontweight='bold',transform=self.histoWidget.axes.transAxes) 
       self.histoWidget.draw()
       sum=0
       print("hist Start")
       for x in range(self.histNumTest1.value(),self.histNumTest2.value()):
         sum=sum+hist1[x]
         print(hist1[x])
       print("hist rst")
       print(sum)
       self.te_histoResult.setText(str(sum[0]))


   def brightChange(self,filepath,var):
      path = os.path.abspath(os.path.join(self.te_dirBright.toPlainText(), os.pardir)) #상위폴더
      image = cv2.imread(filepath, cv2. IMREAD_GRAYSCALE)
      height = image.shape[0]
      width = image.shape[1]

      if var > 0:
         for i in range (0, width):
             for j in range (0, height):
                 if image[i,j]+var > 255:
                     image[i,j] = 255
                     continue
                 image[i,j] += var
      elif var <0:
         for i in range (0, width):
             for j in range (0, height):
                 if image[i,j]+ var < 0:
                     image[i,j] = 0
                     continue
                 image[i,j] += var

      filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
      cv2.imwrite(path+'/brightChange/'+filename+'_(+'+str(var)+').png',image)
      print('brightChange/'+filename+'_(+'+str(var)+').png')


   def scaleChange(self,filepath,var):
      path = os.path.abspath(os.path.join(self.te_dirBright.toPlainText(), os.pardir)) #상위폴더
      image = cv2.imread(filepath ,cv2.IMREAD_GRAYSCALE)     
      scale = var

      height, width = image.shape[:2]
      center_x = int(width / 2) 
      center_y = int(height / 2)
      radius_x, radius_y = int(width / 2), int(height / 2)
      radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)
      min_x, max_x = center_x - radius_x, center_x + radius_x 
      min_y, max_y = center_y - radius_y, center_y + radius_y 
      cropped = image[min_y:max_y, min_x:max_x] 
      new_cropped = cv2.resize(cropped, (width, height)) 

      filename = str(os.path.splitext(os.path.split(filepath)[1])[0])
      cv2.imwrite(path+'/ScaleChange/'+filename+'_('+str(var)+').png',new_cropped)
      print('ScaleChange/'+filename+'_('+str(var)+').png')


   def scaleTest(self):
       image = cv2.imread(self.te_pathScaleTest.toPlainText() ,cv2.IMREAD_GRAYSCALE)     
       scale = self.scaleNumTest.value()
       height, width = image.shape[:2]
       center_x = int(width / 2) 
       center_y = int(height / 2)
       radius_x, radius_y = int(width / 2), int(height / 2)
       radius_x, radius_y = int(scale * radius_x), int(scale * radius_y)

       min_x, max_x = center_x - radius_x, center_x + radius_x 
       min_y, max_y = center_y - radius_y, center_y + radius_y 

       cropped = image[min_y:max_y, min_x:max_x] 
       new_cropped = cv2.resize(cropped, (width, height)) 
       
       cv2.imshow(str(scale)+"zoom", new_cropped)
   
   def tensorSSIM(self, img1path,img2path):
      #print(img1path)
      #print(img2path)
      image1 = tf.io.read_file(img1path)
      image2 = tf.io.read_file(img2path)
      im1 = tf.image.decode_png(image1,channels=1)
      im2 = tf.image.decode_png(image2,channels=1)
      ssim1 = tf.image.ssim(im1, im2, max_val=255, filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.02) # 0< k2 <0.4
      rst = str(ssim1).replace('tf.Tensor','')[1:-1].split(',')
      print("tensorSSIM : "+rst[0])
      return (rst[0])


if  __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Main()
    app.exec_()