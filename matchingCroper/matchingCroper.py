#-*- coding:utf-8 -*-
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, uic , QtGui
import sys
import cv2
import numpy as np
import os
import math
from PIL.ImageQt import ImageQt
from PIL import Image
from PIL import ImageOps

from scipy.misc import toimage
import scipy.misc
from scipy.misc import toimage
import json
from collections import OrderedDict
import tkinter
import tkinter.font
from tkinter import *
import tkinter.filedialog

import time
jsonPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings','settings.json')
matchingMethods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

class Main(QMainWindow):
   def __init__(self):
      super().__init__()   
      formMain=uic.loadUi('ui_main.ui',self)
      self.btn_open1.clicked.connect(self.openDirectory0)
      self.btn_open3.clicked.connect(self.openDirectoryALL)
      self.btn_doAll.clicked.connect(self.doAll)

      self.btn_pattrenMatch.clicked.connect(lambda:self.pattrenMatch(num=1,path = self.te_path1.toPlainText()))
      self.btn_rotate.clicked.connect(lambda:self.rotate(num=1,path=self.te_path1.toPlainText()))
      self.btn_findDegree.clicked.connect(self.findDegree)

      self.btn_pattrenMatch_2.clicked.connect(lambda:self.pattrenMatch(num=2,path=None))
      self.btn_rotate_2.clicked.connect(lambda:self.rotate(num=2, path=None))
      self.btn_findDegree_2.clicked.connect(self.findDegree)

      self.btn_roiSet.clicked.connect(self.roiSet)
      self.btn_roiCut.clicked.connect(self.roiCut)
      self.btn_save.clicked.connect(lambda:self.save(num=1,path=self.te_path1.toPlainText()))

      self.btn_roiSave.clicked.connect(self.roiSave)


      self.show()
      self.loadJson()

   def openDirectory0(self):
      self.folderPath=''
      self.folderPath = (QFileDialog.getOpenFileName(self, "Select file1"))
      print(self.folderPath)
      if self.folderPath is '':
         buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_path1.setText(self.folderPath[0])
         self.lb_img1.setPixmap(QtGui.QPixmap(self.folderPath[0]))
         self.lb_img1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
         self.lb_img1.setScaledContents(True)

   def pattrenMatch(self, num,path):
      if num == 1: 
         img1= cv2.imread(path,cv2.IMREAD_GRAYSCALE)
         dst = cv2.imread(path,cv2.IMREAD_COLOR)
      else : 
         img1=cv2.cvtColor(self.dst, cv2.COLOR_BGR2GRAY)
         dst = self.dst.copy()

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

      # 찾은 템플릿이미지에 흰색 사각형을 쳐줌
      cv2.rectangle(dst, self.top_left1, bottom_right1, 255, 1)
      cv2.rectangle(dst, self.top_left2, bottom_right2, 255, 1)

      cv2.line(dst,self.top_left1,self.top_left2,255,2)
      cv2.line(dst,(0,self.top_left2[1]),(w,self.top_left2[1]),255,2)

      #image = QtGui.QImage(dst.data, w, h, dst.strides[0], QtGui.QImage.Format_Grayscale8)
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)



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

      cv2.rectangle(dst2, self.top_left1, bottom_right1, 255, 1)
      cv2.rectangle(dst2, self.top_left2, bottom_right2, 255, 1)

      cv2.line(dst2,self.top_left1,self.top_left2,255,2)
      cv2.line(dst2,(0,self.top_left2[1]),(w,self.top_left2[1]),255,2)


      self.x1 = self.top_left1[0]+int(self.te_startX.toPlainText())
      self.y1 = self.top_left1[1]+int(self.te_startY.toPlainText())
      self.x2 = self.top_left2[0]+int(self.te_endX.toPlainText())
      self.y2 = self.top_left2[1]+int(self.te_endY.toPlainText())

      self.lb_startX.setText(str(self.x1))
      self.lb_startY.setText(str(self.y1))
      self.lb_endX.setText(str(self.x2))
      self.lb_endY.setText(str(self.y2))

      cv2.rectangle(dst2, (self.x1,self.y1),(self.x2,self.y2), (255,255,255), 2)
      image = QtGui.QImage(dst2.data,dst2.shape[1], dst2.shape[0], dst2.strides[0], QtGui.QImage.Format_RGB888)

      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)

   def roiCut(self):
      self.dstRst=self.dst.copy()
      self.dstRst = self.dst[self.y1:self.y2,self.x1:self.x2]
      #cv2.imshow("test",self.dstRst)
      dst = self.dstRst.copy()
      image = QtGui.QImage(dst.data,dst.shape[1], dst.shape[0], dst.strides[0], QtGui.QImage.Format_RGB888)
      pix = QtGui.QPixmap.fromImage(image)
      self.lb_img2.setPixmap(pix)
      self.lb_img2.setScaledContents(True)

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

   def loadJson(self):       
         try:
            with open(jsonPath, 'r') as f:
               self.jsonData = json.load(f, object_pairs_hook=OrderedDict)
         except Exception as ex:
            self.jsonDataMain=OrderedDict()
            self.jsonData=OrderedDict()
            self.jsonData['startX']=0
            self.jsonData['startY']=0
            self.jsonData['endX']=0
            self.jsonData['endY']=0
            self.jsonDataMain["Option"]=self.jsonData
            with open(jsonPath, 'w') as make_file:
               json.dump(self.jsonDataMain, make_file,ensure_ascii=False,indent=4)
            with open(jsonPath, 'r') as f:
               self.jsonData = json.load(f, object_pairs_hook=OrderedDict)
           
         self.te_startX.setText(str(self.jsonData['Option']['startX']))
         self.te_startY.setText(str(self.jsonData['Option']['startY']))
         self.te_endX.setText(str(self.jsonData['Option']['endX']))
         self.te_endY.setText(str(self.jsonData['Option']['endY']))

   def roiSave(self):
         print(self.jsonData.keys())
         self.jsonData['Option']['startX']=self.te_startX.toPlainText()
         self.jsonData['Option']['startY']=self.te_startY.toPlainText()
         self.jsonData['Option']['endX']=self.te_endX.toPlainText()
         self.jsonData['Option']['endY']=self.te_endY.toPlainText()
         with open(jsonPath, 'w') as outfile:
            json.dump(self.jsonData, outfile,indent=4)
         QMessageBox.information(self, 'PyQt5 message', 'Settings Save Complete', QMessageBox.Ok )

   def openDirectoryALL(self):
      self.dirPath=''
      self.dirPath = (QFileDialog.getExistingDirectory(self, "Select dir"))
      print(self.dirPath)
      if self.dirPath is '':
         buttonReply = QMessageBox.information(self, 'PyQt5 message', "Please choose Directory", QMessageBox.Ok )
      else:
         self.te_path3.setText(self.dirPath)
         self.fileList = []
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


if  __name__ == "__main__":
    app = QApplication(sys.argv)
    main = Main()
    app.exec_()