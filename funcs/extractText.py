#!usr/bin/python

import caffe
import numpy as np
import collections
import classifier
import os
import sklearn
import scipy
from scipy.ndimage import filters
from sklearn.feature_extraction import image

def extractTextBoxes(A,scales):
	# Extract info from image
	#A = np.lib.pad(A, ((scale1/4,scale1/4), (scale2/4,scale2/4), (0,0)), 'edge')
	#H, W, C = A.shape
	# Scales to work with: 24x24, 32x32, 40x40, 48x48
	#scales = [220]
	#scales = [15,20]
	#scl = 32
	# Strides to make along the horizontal and vertical directions
	stride = 4
	thresh = 0.7 # To prune truly detected Char containing windows
	out = collections.defaultdict(int) # Character Detection Scores
	#inxI = collections.defaultdict(int) # X of Current Sliding Window
	#inxJ = collections.defaultdict(int) # Y of Current Sliding Window
	PROJECT_ROOT = "/Users/aditya/Desktop/quarter5/CS231N/project/text_det/cs231n/"
	TRAIN_ROOT = os.path.join(PROJECT_ROOT, "train")
	SNAPSHOT_ROOT = os.path.join(TRAIN_ROOT, "snapshots")
	DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
	print "CAFFE_ROOT=%s" % PROJECT_ROOT
	print "TRAIN_ROOT=%s" % TRAIN_ROOT
	print "SNAPSHOT_ROOT=%s" % SNAPSHOT_ROOT

	RECOGNIZER_MODEL_FILE = os.path.join(os.path.join(TRAIN_ROOT, "mnist"), "char74k_recognizer_deploy.prototxt")
	RECOGNIZER_PRETRAINED = os.path.join(SNAPSHOT_ROOT, "recognizer_iter_15000.caffemodel")

	DETECTOR_MODEL_FILE = os.path.join(os.path.join(TRAIN_ROOT, "mnist"), "char74k_detector_deploy.prototxt")
	DETECTOR_PRETRAINED = os.path.join(SNAPSHOT_ROOT, "detector_iter_15000.caffemodel")

	recognizer = classifier.Recognizer(RECOGNIZER_MODEL_FILE, RECOGNIZER_PRETRAINED)
	detector = classifier.Detector(DETECTOR_MODEL_FILE, DETECTOR_PRETRAINED)
	for scl in xrange(len(scales)):
		scale1, scale2 = scales[scl]
		#A = np.lib.pad(A, ((scale1/4,scale1/4), (scale2/4,scale2/4), (0,0)), 'edge')
		H, W, C = A.shape
		#resizedA = scipy.misc.imresize(A,imgScale,interp='bicubic')
		patches = sklearn.feature_extraction.image.extract_patches(A, patch_shape=[scale1,scale2,C],extraction_step = stride)
		#patches = np.squeeze(patches)
		print patches.shape
		hExtent,wExtent,_,_,_,_ = patches.shape
		#currOut = np.zeros((hExtent,wExtent))
		#inxCurrI = np.zeros((hExtent,wExtent))
		#nxCurrJ = np.zeros((hExtent,wExtent))
		#outX = (detector.detect(patches.reshape(hExtent*wExtent,scl,scl,C)))[:,1]
		outX = np.amax(recognizer.recognize(patches.reshape(hExtent*wExtent,scale1,scale2,C)),axis=1)
		#outY = (detector.detect(patches.reshape(hExtent*wExtent,scl,scl,C))[:,1])
		#outX = outX*outY
		currOut = outX.reshape(hExtent,wExtent)
		out[scl] = currOut
		print 'Done extracting scores for scale ('+str(scale1)+','+str(scale2)+')'
		#print np.amax(currOut)
		#print currOut

		#for hPatch in np.arange(hExtent):
		#	print 'Running detection for row '+str(hPatch)+'/'+str(hExtent)
		#	currRowPatches = patches[hPatch]
		#	currOut[hPatch] = (detector.detect(currRowPatches)[:,1])
		#	inxCurrI[hPatch,:] = hPatch*stride*np.ones(wExtent)
		#	inxCurrJ[hPatch,:] = stride*np.arange(wExtent)

		#out[scl] = currOut
		#inxI[scl] = inxCurrI
		#inxJ[scl] = inxCurrJ
		#print 'Done extracting scores for scale '+str(scl)

	delta = 1 # Number of pixels on either side to compare for NMS
	#supOut = collections.defaultdict(int) # Suppressed outputs after NMS, in each row (Rs)
	#nonZeroRows = collections.defaultdict(int) # Sets containing Row indices with non-zero Rs
	boundingBoxes = collections.defaultdict(int) # Dict of 5-tuples, scale, xmin, xmax, xarray, rowIdx, score
	bbIdx = 1
	print 'Will now start doing row-wise NMS with delta = '+str(delta)+' ...'
	for scl in xrange(len(scales)):
		scores = out[scl]
		#yInx = inxI[scl]
		#xInx = inxJ[scl]
		HH, WW = scores.shape
		#suppressed = np.zeros_like(scores)
		#nnzRows = set()
		#interMed = scipy.ndimage.filters.maximum_filter(scores,size=(1,1+2*delta))
		#mask = (interMed == scores) and (scores > thresh)

		for y in np.arange(HH):#[:-scl]:
			currMin = 0
			currMax = 0
			firstEnc = 0
			avScore = 0.0 # Score for current bounding box = avg of nms output
			avCount = 0 # Num of non zero elements in nms output
			flagToAppend = 0 # Whether or not to append current row to list of BBs
			bbArray = list() # Array containing x indices of peaks along current row
			for x in np.arange(WW)[delta:-delta]:
				maxCurr = np.amax(scores[y,x-delta:x+delta])
				if scores[y,x]==maxCurr and maxCurr>thresh: # To make sure that the detection is a char
					#suppressed[y,x] = maxCurr
					#nnzRows.add(y)
					currMax = x*stride
					avCount += 1
					avScore += maxCurr
					flagToAppend = 1
					bbArray.append(x*stride)
					if firstEnc==0:
						currMin = x*stride
						firstEnc = 1
				#else:
				#	suppressed[y,x] = 0
			if firstEnc==1:
				avScore = avScore/avCount
				currTuple = scl, currMin, currMax, np.asarray(bbArray), y*stride, avScore
				boundingBoxes[bbIdx] = currTuple
				bbIdx += 1
			#boundingBoxes.append(currTuple)

		#supOut[scl] = suppressed
		#nonZeroRows[scl] = nnzRows
		print 'Done finding bounding boxes for scale '+str(scales[scl])

	print 'Going to start pruning the bounding boxes now...'
	print 'Num of boxes before pruning is '+str(bbIdx-1)
	toBeRemoved = set() # List containing the dictionary indices of keys to be removed
	print boundingBoxes.keys()
	for tpl1 in boundingBoxes.keys():
		scl1, x11, x12, xArr1, y11, score1 = boundingBoxes[tpl1]
		y12 = y11 + scales[scl1][0]
		x12 = x12 + scales[scl1][1]
		area1 = np.abs((x12-x11)*(y11-y12))
		for tpl2 in boundingBoxes.keys():
			scl2, x21, x22, xArr2, y21, score2 = boundingBoxes[tpl2]
			y22 = y21 + scales[scl2][0]
			x22 = x22 + scales[scl2][1]
			area2 = np.abs((x22-x21)*(y22-y21))
			xOverlap = np.amax((0, np.amin((x12,x22)) - np.amax((x11,x21)) ))
			yOverlap = np.amax((0, np.amin((y12,y22)) - np.amax((y11,y21)) ))
			intArea = xOverlap*yOverlap
			if tpl1!=tpl2 and (float(intArea)/np.amin((area1,area2)) > 0.5):
				if score1>score2:
					toBeRemoved.add(tpl2)
				else:
					toBeRemoved.add(tpl1)

	print 'Going to prune:'
	print toBeRemoved
	for inx in toBeRemoved:
		boundingBoxes.pop(inx)


	# Find the text corresponding to the surviving bounding boxes
	#numToChar = collections.defaultdict(int) # Dictionary mapping class numbers to labels
	print 'Pruning bounding boxes completed. Will now run recognizer on the bounding boxes...'
	print 'Remaining boxes:'+str(boundingBoxes.keys())
	# Going to remove repeated matches for the same letter
	for inx in boundingBoxes.keys():
		scl, xMin, xMax, xArr, y, score = boundingBoxes[inx]
		scale1, scale2 = scales[scl]
		prevXIdx = -1000
		currScores = out[scl]
		numDistinctLetters = 0
		firstPeak = 1
		xNewArr = list()
		print 'Current array being processed: '+str(xArr)
		for xPeak in xArr:
			if np.abs(xPeak - prevXIdx) <= scale2/2 and firstPeak==0:
				currIdxs.append(xPeak)
				xScore = out[y,xPeak]
				currScores.append(xScore)
			if np.abs(xPeak - prevXIdx) > scale2/2 and firstPeak==0:
				peakToRetain = np.asarray(currIdxs)[np.argmax(np.asarray(currScores))]
				xNewArr.append(peakToRetain)
				currIdxs = list()
				currScores = list()
				currIdxs.append(xPeak)
				currScores.append(out[y,xPeak])
			if firstPeak==1:
				currIdxs = list()
				currScores = list()
				currIdxs.append(xPeak)
				currScores.append(out[y,xPeak])
				firstPeak = 0
			if xPeak==xArr[-1]:
				peakToRetain = np.asarray(currIdxs)[np.argmax(np.asarray(currScores))]
				xNewArr.append(peakToRetain)
			print 'Finished processing '+str(xPeak)
			prevXIdx = xPeak
		boundingBoxes[inx] = scl, xNewArr, y, score


	predictions = collections.defaultdict(int)
	predCount = 1
	for inx in boundingBoxes.keys():
		scl, xArr, y, score = boundingBoxes[inx]
		#xMin = np.amin(xArr)
		#xMax = np.amax(xArr)
		print 'Scale of box '+str(predCount)+ ' is '+str(scales[scl])
		currText = list()
		charNum = 0
		#xPrunedArr = list()
		#for i in xArr.shape[0]:
		#	if (i-1)>=0 and np.abs(xArr[i-1]-xArr[i])

		for xPeak in xArr:
			currPatch = A[y:y+scales[scl][0], xPeak:xPeak+scales[scl][1]]
			preds = (recognizer.recognize([currPatch])[0])
			label = classifier.transform_label(np.argmax(preds))
			currText.append(label)

		predictions[inx] = ''.join(currText)
		predCount += 1

	print 'The array of x indices with a char is:'
	print xArr
	predCount = 0
	for inx in predictions.keys():
		print 'bounding box '+str(predCount)+' :'
		print predictions[inx]
		predCount += 1

	print 'Number of rows predicted for: '+str(predCount)


# Decide the scales of different sliding windows
#imgFile = '../data/svt1/img/00_07.jpg'
imgFile = "/Users/aditya/Downloads/word/word/1/10.jpg"
A = caffe.io.load_image(imgFile)
scales = [(220,220)]
extractTextBoxes(A,scales)

# Function to be called: giveMeScore(A)
# Function to be called: giveMeChar(A)