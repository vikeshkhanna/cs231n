#!/bin/bash
# Takes a LISTFILE for convert_imagese, shuffles it randomly and splits into train and test parts.

RESIZE=64

if [ "$#" -ne 2 ]; then
  echo "Usage: ./script.sh LISTFILE.txt <no_of_lines_in_test>"
  exit 1
fi

LISTFILE="$1"
LINES="$2"

cat $LISTFILE | gshuf > "$LISTFILE.shuffle"
cat "$LISTFILE.shuffle" | head -n "$LINES" > "$LISTFILE.test"
cat "$LISTFILE.shuffle" | tail -n +"$LINES" > "$LISTFILE.train"

# Delete existing databases
echo "DELETING EXISTING char_74k dataset"
rm -rf char74k_lmdb_*

# Call ./convert_imageset
echo "Converting train images..."
$CAFFE_ROOT/build/tools/convert_imageset -resize_height=$RESIZE -resize_width=$RESIZE English/ "$LISTFILE.train" char74k_lmdb_train
echo "Converting test images..."
$CAFFE_ROOT/build/tools/convert_imageset -resize_height=$RESIZE -resize_width=$RESIZE English/ "$LISTFILE.test" char74k_lmdb_test
