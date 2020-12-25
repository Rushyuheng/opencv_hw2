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
		#self.iniGuiEvent()

'''
	def iniGuiEvent(self):# connect all button to all event slot
		self.pushButton_showImg.clicked.connect(self.pushButton_showImg_onClick)
		self.pushButton_ShowPara.clicked.connect(self.pushButton_ShowPara_onClick)
		self.pushButton_ShowStruct.clicked.connect(self.pushButton_ShowStruct_onClick)
		self.pushButton_ShowAcc.clicked.connect(self.pushButton_ShowAcc_onClick)
		self.pushButton_Test.clicked.connect(self.pushButton_Test_onClick)

	#5.1 show data image
	@QtCore.pyqtSlot()
	def pushButton_showImg_onClick(self):
		batch1 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_1')
		batch2 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_2')
		batch3 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_3')
		batch4 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_4')
		batch5 = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/data_batch_5')

		batchlist = [batch1,batch2,batch3,batch4,batch5]
		imglist = []
		labellist = []
		for i in range(10):
			index = random.randint(0,9999)
			batchindex = random.randint(0,4)
			img = np.transpose(np.reshape(batchlist[batchindex][b'data'][index],(3, 32,32)), (1,2,0))
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #convert color channel
			imglist.append(img)
			label = batchlist[batchindex][b'labels'][index]
			labellist.append(label)

		self.w = loadimg.AnotherWindow(imglist,labellist)
		self.w.show()

	#5.2 show hyperparameter structure
	@QtCore.pyqtSlot()
	def pushButton_ShowPara_onClick(self):
		batchsize = 32
		learningrate = 0.001
		print('hyperparameter:')
		print('batchsize: %d' %batchsize)
		print('learning rate: %.3f' %learningrate)
		print('optimizer: SGD')


	#5.3 show model structure
	@QtCore.pyqtSlot()
	def pushButton_ShowStruct_onClick(self):
		self.model.summary()

	#5.4 show train history
	@QtCore.pyqtSlot()
	def pushButton_ShowAcc_onClick(self):
		acc = cv2.imread('./img/acc.png')
		cv2.imshow('Accuracy',acc)
		loss = cv2.imread('./img/loss.png')
		cv2.imshow('Loss',loss)

	#5.5 test
	@QtCore.pyqtSlot()
	def pushButton_Test_onClick(self):
		plt.close('all')
		testbatch = ld.load_data_set('./cifar-10-python/cifar-10-batches-py/test_batch')
		index = int(self.ImgIndex.text())
		img = np.transpose(np.reshape(testbatch[b'data'][index],(3,32,32)), (1,2,0))
		simg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		#show test img
		res = cv2.resize(simg, dsize=(128, 128), interpolation = cv2.INTER_CUBIC)
		cv2.imshow('test image',res)

		img = np.expand_dims(img, axis=0)
		predictions = self.model.predict(img.astype('float16'))
		category = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'] # label to name
		plt.bar(category,predictions[0])
		plt.show()
'''		

if __name__ == "__main__": #main function
	def run_app():
		app = QtWidgets.QApplication(sys.argv)
		window = MainUi()
		window.show()
		app.exec_()
	run_app()