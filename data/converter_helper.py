'''
Creates the LISTFILE used as input to the convert_imageset tool of caffe
'''

import sys, os

if len(sys.argv) < 3:
  print ("Insufficient arguments. Usage : python prog.py <Root> <MODE>.\nMODE=Img, Fnt, Hnd or All\nRoot folder must contain the MODE folder (except All)")
  sys.exit(1)

ROOT = sys.argv[1]
MODE = sys.argv[2]
# Labes are deterministic - In the order [0-9][A-Z][a-z]
FOLDER_PREFIX = "Sample0"
# Maps folder name to label
LABELS = {}

cnt = 1

# Labels must be sequentially numbered, starting from zero
for i in range(10):
  if cnt < 10:
    LABELS[FOLDER_PREFIX + "0" + str(cnt)] = cnt-1
  else:
    LABELS[FOLDER_PREFIX + str(cnt)] = cnt-1

  cnt += 1

for i in range(26):
  LABELS[FOLDER_PREFIX + str(cnt)] = cnt - 1
  cnt += 1

for i in range(26):
  LABELS[FOLDER_PREFIX + str(cnt)] = cnt - 1
  cnt += 1

subdirs = [MODE]

if MODE.lower() == "all":
  subdirs = ["Img", "Fnt", "Hnd"]

for subdir in subdirs:
  relative = subdir

  # Fix for Img subdir
  if subdir.lower() == "img":
    relative = os.path.join (relative, "GoodImg")
    relative = os.path.join (relative, "Bmp")
  elif subdir.lower() == "hnd":
    relative = os.path.join (relative, "Img")

  for folder_name, label in LABELS.iteritems():
    relative_folder = os.path.join(relative, folder_name)
    folder = os.path.join(ROOT, relative_folder)

    for fl in os.listdir(folder):
      if fl.endswith(".png"):
        relative_path = os.path.join(relative_folder, fl)
        print("%s %s" % (relative_path, label))
