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
		self.model = models.load_model('./my_model.h5')#load model


	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_DrawContour.clicked.connect(self.pushButton_DrawContour_onClick)
		self.pushButton_CountCoin.clicked.connect(self.pushButton_CountCoin_onClick)
		self.pushButton_FindCor.clicked.connect(self.pushButton_FindCor_onClick)
		self.pushButton_intr.clicked.connect(self.pushButton_intr_onClick)
		self.pushButton_extr.clicked.connect(self.pushButton_extr_onClick)
		self.pushButton_Dist.clicked.connect(self.pushButton_Dist_onClick)
		self.pushButton_AR.clicked.connect(self.pushButton_AR_onClick)
		self.pushButton_DisMap.clicked.connect(self.pushButton_DisMap_onClick)
		self.pushButton_Train.clicked.connect(self.pushButton_Train_onClick)
		self.pushButton_test.clicked.connect(self.pushButton_test_onClick)
		self.pushButton_TensorB.clicked.connect(self.pushButton_TensorB_onClick)
		self.pushButton_DataAug.clicked.connect(self.pushButton_DataAug_onClick)

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
			ret = []
			corners = []
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

	#3 AR
	@QtCore.pyqtSlot()
	def pushButton_AR_onClick(self):
		cv2.destroyAllWindows()
		pyramid = [[3,3,-3], [1,1,0],[3,5,0],[5,1,0]]
		pyramid = np.array(pyramid, dtype=np.float32)

		#calibrate camera first
		objp = np.zeros((8*11,3), np.float32)
		objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)

		for i in range(1,6): # use bmp 1-5
			objpoints = [] # 3d point in real world space
			imgpoints = [] # 2d points in image plane.	
			filename = './Datasets/Q3_Image/' + str(i) + '.bmp'
			img = cv2.imread(filename)
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (11,8),None)
			if ret == True:
				objpoints.append(objp)
				imgpoints.append(corners)

			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

			#do projection
			twodpro, jacobian = cv2.projectPoints(pyramid, rvecs[0], tvecs[0], mtx, dist)

			#draw it on img

			img	= cv2.line(img, tuple(twodpro[0][0]), tuple(twodpro[1][0]), (0, 0, 255), 5)
			img	= cv2.line(img, tuple(twodpro[0][0]), tuple(twodpro[2][0]), (0, 0, 255), 5)
			img	= cv2.line(img, tuple(twodpro[0][0]), tuple(twodpro[3][0]), (0, 0, 255), 5)
			#triangle
			img	= cv2.line(img, tuple(twodpro[1][0]), tuple(twodpro[2][0]), (0, 0, 255), 5)
			img	= cv2.line(img, tuple(twodpro[2][0]), tuple(twodpro[3][0]), (0, 0, 255), 5)
			img	= cv2.line(img, tuple(twodpro[3][0]), tuple(twodpro[1][0]), (0, 0, 255), 5)
			img = cv2.resize(img, (1024, 1024))  
			cv2.imshow('result',img)
			cv2.waitKey(500)

	#4 disparity
	#mouse click event
	def mouse(self,event,x,y,flags,param):
		if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
			normdis = param
			grayvalue = normdis[y,x,0]
			cv2.rectangle(normdis, (505, 380), (705, 480), (255, 255, 255), -1)
			baseline = 178
			flen = 2826
			cx = 123
			depth = (baseline * flen) / (grayvalue + cx)
			text1 = 'disparity = {value} pixels'.format(value = grayvalue.astype(int))
			text2 = 'depth = {value} mm'.format(value = depth.astype(int))
			cv2.putText(normdis, text1, (505, 400), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)
			cv2.putText(normdis, text2, (505, 440), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1)


	@QtCore.pyqtSlot()
	def pushButton_DisMap_onClick(self):
		cv2.destroyAllWindows()	
		imgL = cv2.imread('./Datasets/Q4_Image/imgL.png')
		imgR = cv2.imread('./Datasets/Q4_Image/imgR.png')
		imgL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
		imgR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

		stereo = cv2.StereoBM_create(numDisparities=256, blockSize=21)
		disparity = stereo.compute(imgL,imgR)
		disparity = abs(disparity)
		normdis = disparity / disparity.max() * 255 # normalize
		normdis  = normdis.astype('uint8')
		normdis = cv2.resize(normdis, (705, 480))#1410 960
		normdis = cv2.cvtColor(normdis,cv2.COLOR_GRAY2BGR)
		cv2.namedWindow('disparity map')
		cv2.setMouseCallback('disparity map',self.mouse,param = normdis)

		while(1):
			cv2.imshow('disparity map',normdis)
			if cv2.waitKey(20) & 0xFF == 27:
				break

		#if esc pressed, finish.
		cv2.destroyAllWindows()

	# 5.1 ResNet50
	@QtCore.pyqtSlot()
	def pushButton_Train_onClick(self):
		cv2.destroyAllWindows()
		result = cv2.imread('./trainacc.png')
		cv2.imshow('train accuracy',result)
		
	# 5.2 tensorboard
	@QtCore.pyqtSlot()
	def pushButton_TensorB_onClick(self):
		cv2.destroyAllWindows()
		result = cv2.imread('./tensorboardresult.png')
		cv2.imshow('tensorboard result',result)

	# 5.3 testing
	@QtCore.pyqtSlot()
	def pushButton_test_onClick(self):
		cv2.destroyAllWindows()
		index = random.randint(0,12500)
		cata = random.choice(['Cat/','Dog/'])
		img = cv2.imread('./Datasets/PetImages/' + cata + str(index) + '.jpg')
		testimg = cv2.resize(img, dsize=(224, 224))
		testimg = cv2.normalize(testimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		testimg = cv2.cvtColor(testimg, cv2.COLOR_RGB2BGR)
		testimg = np.expand_dims(testimg, axis=0)

		predictions = self.model.predict(testimg.astype('float32')) #normalize
		category = ['Cat','Dog'] # label to name
		#show test img and prediction
		res = cv2.resize(img, dsize=(224, 224))
		cv2.imshow('predictions = ' + category[np.argmax(predictions)],res)

	# 5.4 data argument
	@QtCore.pyqtSlot()
	def pushButton_DataAug_onClick(self):
		cv2.destroyAllWindows()
		result = cv2.imread('./aug.png')
		cv2.imshow('data argument',result)

		

		
if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()