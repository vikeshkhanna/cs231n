import numpy as np
import caffe

def transform_label (caffe_label):
    if caffe_label >=0 and caffe_label <=9:
        return str(caffe_label)
    elif caffe_label >=10 and caffe_label <= 35:
        return chr(ord('A') + caffe_label - 10)
    elif caffe_label >=36 and caffe_label <= 61:
        return chr(ord('a') + caffe_label - 36)
    raise "Invalid label from Caffe"

class Classifier:
  def __init__(self, model_file, pretrained_file, gpu=False):
    self.MODEL_FILE = model_file
    self.PRETRAINED = pretrained_file
    
    if gpu:
      caffe.set_mode_gpu()
    else:
      caffe.set_mode_cpu()

    self.classifier = caffe.Classifier(self.MODEL_FILE, self.PRETRAINED)

class Recognizer(Classifier):
    # Recognizes a list of images
  def recognize(self, np_images, oversample=False):
    return self.classifier.predict(np_images, oversample=oversample)

class Detector(Classifier):
  # Recognizes a list of images
  def detect(self, np_images, oversample=False):
    return self.classifier.predict(np_images, oversample=oversample)
