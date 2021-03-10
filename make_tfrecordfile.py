import os
import cv2
import random
import tensorflow as tf


dataset_dir = "data"
flowers = ["faces"]
tfrecord_train = "faces_train.tfrecord"
photo_files = []

def convert_dataset(filenames, tfrecord_file):
    with tf.python_io.TFRecordWriter(tfrecord_file) as tfrecord_writer:
        for file in filenames:
            img = cv2.imread(file)
            img = cv2.resize(img, (28, 28))
            example = tf.train.Example(features=tf.train.Features(
                feature={"image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                         "filename": tf.compat.v1.train.Feature(bytes_list=tf.compat.v1.train.BytesList(value=[file.encode()]))}))
            print(file, flowers.index(os.path.basename(os.path.dirname(file))))
            tfrecord_writer.write(example.SerializeToString())

for flower in flowers:
    for image in os.listdir(os.path.join(dataset_dir, flower)):
        photo_files.append(os.path.join(dataset_dir, os.path.join(flower, image)))
random.shuffle(photo_files)
convert_dataset(photo_files, tfrecord_train)
print("train:", len(photo_files))