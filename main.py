import sys
import cv2
import numpy as np
import pickle
from PyQt5 import QtGui, QtCore, QtWidgets, uic
import random
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib


class MainUi(QtWidgets.QMainWindow):
	def __init__(self):
		QtWidgets.QMainWindow.__init__(self)
		uic.loadUi('./ui/main.ui', self)
		self.iniGuiEvent()


	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_DrawContour.clicked.connect(self.pushButton_DrawContour_onClick)
		self.pushButton_CountCoin.clicked.connect(self.pushButton_CountCoin_onClick)
		self.pushButton_FindCor.clicked.connect(self.pushButton_FindCor_onClick)
		self.pushButton_intr.clicked.connect(self.pushButton_intr_onClick)
		self.pushButton_extr.clicked.connect(self.pushButton_extr_onClick)
		self.pushButton_Dist.clicked.connect(self.pushButton_Dist_onClick)

	#1.1 draw contour
	@QtCore.pyqtSlot()
	def pushButton_DrawContour_onClick(self):
		cv2.destroyAllWindows()
		coin01 = cv2.imread('./Datasets/Q1_Image/coin01.jpg')
		coin02 = cv2.imread('./Datasets/Q1_Image/coin02.jpg')

		coin01_g = cv2.cvtColor(coin01, cv2.COLOR_BGR2GRAY)
		coin02_g = cv2.cvtColor(coin02, cv2.COLOR_BGR2GRAY)

		coin01_blur = cv2.GaussianBlur(coin01_g, (11, 11), 0)
		coin02_blur = cv2.GaussianBlur(coin02_g, (11, 11), 0)

		coin01_bin = cv2.Canny(coin01_blur, 20, 160)
		coin02_bin = cv2.Canny(coin02_blur, 20, 160)

		self.contours1, hierarchy1 = cv2.findContours(coin01_bin ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		self.contours2, hierarchy2 = cv2.findContours(coin02_bin ,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

		cv2.drawContours(coin01,self.contours1,-1,(0,0,255),3)
		cv2.drawContours(coin02,self.contours2,-1,(0,0,255),3)

		cv2.imshow('coin1',coin01)
		cv2.imshow('coin2',coin02)

	#1.2 count contour
	@QtCore.pyqtSlot()
	def pushButton_CountCoin_onClick(self):
		self.label_Coin1.setText('There are {num} coins '.format(num = len(self.contours1)))
		self.label_Coin2.setText('There are {num} coins '.format(num = len(self.contours2)))

	#2.1 find corner
	@QtCore.pyqtSlot()
	def pushButton_FindCor_onClick(self):
		cv2.destroyAllWindows()
		for i in range(1,16):
			filename = './Datasets/Q2_Image/' + str(i) + '.bmp'
			img = cv2.imread(filename)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
			img = cv2.drawChessboardCorners(img, (11,8), corners,ret)

			img = cv2.resize(img, (1024, 1024))    
			cv2.imshow(str(i) + '.bmp',img)

	#2.2 find intrinsic
	@QtCore.pyqtSlot()
	def pushButton_intr_onClick(self):
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.	

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((8*11,3), np.float32)
		objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
		for i in range(1,16):
			filename = './Datasets/Q2_Image/' + str(i) + '.bmp'
			img = cv2.imread(filename)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
			if ret == True:
				objpoints.append(objp)
				imgpoints.append(corners)

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		print('Intrinsic matrix:')
		print(mtx)



	#2.3 find extrinsic
	@QtCore.pyqtSlot()
	def pushButton_extr_onClick(self):
		text = str(self.comboBox.currentText())
		index = int(text.split(".")[0])

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((8*11,3), np.float32)
		objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.	
		filename = './Datasets/Q2_Image/' + str(index) + '.bmp'
		img = cv2.imread(filename)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None) # seperate extrmat
		rvecs = np.array(rvecs, dtype=np.float32)
		tvecs = np.array(tvecs, dtype=np.float32)[0] #regularize dim
		rmat , jacob= cv2.Rodrigues(rvecs)
		extmat = np.hstack((rmat,tvecs))

		print(str(index) + '.bmp\'s extrinsic matrix:')
		print(extmat)

	#2.4 find distortion
	@QtCore.pyqtSlot()
	def pushButton_Dist_onClick(self):
		objpoints = [] # 3d point in real world space
		imgpoints = [] # 2d points in image plane.	

		# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
		objp = np.zeros((8*11,3), np.float32)
		objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
		for i in range(1,16):
			filename = './Datasets/Q2_Image/' + str(i) + '.bmp'
			img = cv2.imread(filename)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
			if ret == True:
				objpoints.append(objp)
				imgpoints.append(corners)

		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
		print('Distortion matrix:')
		print(dist)

if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()