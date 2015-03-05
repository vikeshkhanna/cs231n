#!usr/bin/python

import cv2
import numpy as np
import collections

def singleImage(A):
	# Extract info from image
	H, W, C = A.shape
	# Scales to work with: 24x24, 32x32, 40x40, 48x48
	scales = [24, 32, 40, 48]
	# Strides to make along the horizontal and vertical directions
	horStride = 4
	verStride = 4
	out = collections.defaultdict('int') # Character Detection Scores
	inxI = collections.defaultdict('int') # X of Current Sliding Window
	inxJ = collections.defaultdict('int') # Y of Current Sliding Window
	for scl in scales:
		HH = scl
		WW = scl
		hRn = xrange(H-HH+1)
		wRn = xrange(W-WW+1)
		xId = 0
		yId = 0
		currOut = np.zeros((1 + (H - HH)/verStride, 1 + (W - WW)/horStride))
		inxCurrI = np.zeros((1 + (H - HH)/verStride, 1 + (W - WW)/horStride))
		inxCurrJ = np.zeros((1 + (H - HH)/verStride, 1 + (W - WW)/horStride))
		for ix in hRn[0::verStride]:
			xId += 1
			for jx in wRn[0::horStride]:
				yId += 1
				currPatch = A[ix:ix+HH, jx:jx+WW]
				currOut[xId, yId] = giveMeScore(currPatch)
				inxCurrI[xId, yId] = ix
				inxCurrJ[xId, yId] = jx

		out[scl] = currOut
		inxI[scl] = inxCurrI
		inxJ[scl] = inxCurrJ

	delta = 3 # Number of pixels on either side to compare for NMS
	supOut = collections.defaultdict(int) # Suppressed outputs after NMS, in each row (Rs)
	nonZeroRows = collections.defaultdict(int) # Sets containing Row indices with non-zero Rs
	boundingBoxes = collections.defaultdict(int) # Dict of 5-tuples, scale, xmin, xmax, xarray, rowIdx, score
	bbIdx = 1
	for scl in scales:
		scores = out[scl]
		yInx = inxI[scl]
		xInx = inxJ[scl]
		HH, WW = scores.shape
		suppressed = np.zeros_like(scores)
		nnzRows = set()
		for y in np.arange(HH)[:-scl]:
			currMin = 0
			currMax = 0
			firstEnc = 0
			avScore = 0.0 # Score for current bounding box = avg of nms output
			avCount = 0 # Num of non zero elements in nms output
			flagToAppend = 0 # Whether or not to append current row to list of BBs
			bbArray = list() # Array containing x indices of peaks along current row
			for x in np.arange(WW)[delta:-delta]:
				maxCurr = np.amax(scores[y,x-delta:x+delta])
				if scores[y,x]==maxCurr:
					suppressed[y,x] = maxCurr
					nnzRows.add(y)
					currMax = xInx[y,x]
					avCount += 1
					avScore += maxCurr
					flagToAppend = 1
					bbArray.append(xInx[y,x])
					if firstEnc==0:
						currMin = xInx[y,x]
						firstEnc = 1
				else:
					suppressed[y,x] = 0
			if firstEnc==1:
				avScore = avScore/avCount
				currTuple = scl, currMin, currMax, np.asarray(bbArray), yInx[y,x], avScore
				boundingBoxes[bbIdx] = currTuple
				bbIdx += 1
			#boundingBoxes.append(currTuple)

		supOut[scl] = suppressed
		nonZeroRows[scl] = nnzRows

	toBeRemoved = list() # List containing the dictionary indices of keys to be removed
	for tpl1 in boundingBoxes.keys():
		scl1, x11, x12, xArr1, y11, score1 = boundingBoxes[tpl1]
		y12 = y11 + scl1
		area1 = np.abs((x12-x11)*(y11-y12))
		for tpl2 in boundingBoxes.keys():
			scl2, x21, x22, xArr2, y21, score2 = boundingBoxes[tpl2]
			y22 = y21 + scl2
			area2 = np.abs((x22-x21)*(y22-y21))
			xOverlap = np.amax((0, np.amin((x12,x22)) - np.amax((x11,x21)) ))
			yOverlap = np.amax((0, np.amin((y12,y22)) - np.amax((y11,y21)) ))
			intArea = xOverlap*yOverlap
			if (float(intArea)/np.amin((area1,area2)) > 0.5):
				if score1>score2:
					toBeRemoved.append(tpl2)
				else:
					toBeRemoved.append(tpl1)

	for inx in toBeRemoved:
		boundingBoxes.pop(inx)


	# Find the text corresponding to the surviving bounding boxes
	numToChar = collections.defaultdict(int) # Dictionary mapping class numbers to labels
	predictions = collections.defaultdict(int)
	predCount = 1
	for inx in boundingBoxes.keys():
		scl, xMin, xMax, xArr, y, score = boundingBoxes[inx]
		currText = list()
		charNum = 0
		for xPeak in xArr:
			currText.append(numToChar[giveMeChar(A[y:y+scl, xPeak:xPeak+scl])])

		predictions[inx] = ''.join(currText)
		predCount += 1

	predCount = 1
	for inx in predictions.keys():
		print 'bounding box '+str(predCount)+' :'
		print predictions[inx]
		predCount += 1

	print 'Number of rows predicted for: '+str(predCount)


# Decide the scales of different sliding windows



# Function to be called: giveMeScore(A)
# Function to be called: giveMeChar(A)