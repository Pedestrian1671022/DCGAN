import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Training Params
image_pixels = 96
epochs = 400
batch_size = 40
train_size = 16412
train_tfrecord = "faces_train.tfrecord"

#Mapping function
def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"image":tf.FixedLenFeature([], tf.string),
                                                                     "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, filename

# Start training
with tf.Session() as sess:
    dataset_train = tf.data.TFRecordDataset([train_tfrecord])
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(400).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)

    for epoch in range(epochs):
        for step in range(int(train_size/batch_size)):
            batch_x, _ = sess.run(next_element_train)
            batch_x = batch_x/255.
            f, a = plt.subplots(4, 10, figsize=(10, 4))
            cnt = 0
            for i in range(10):
                for j in range(4):
                    a[j][i].imshow(batch_x[cnt, :, :, ::-1])
                    cnt += 1
            f.savefig("faces_original/faces_%d_%d.png" % (epoch, step))
            plt.close()
