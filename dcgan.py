import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training Params
image_pixels = 96
epochs = 400
batch_size = 40
train_size = 16412
train_tfrecord = "faces_train.tfrecord"

# Network Params
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])
    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, filename


# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6 * 6 * 64)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, units=24 * 24 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 24, 24, 128])
        # Deconvolution, image shape: (batch, 48, 48, 32)
        x = tf.layers.conv2d_transpose(x, 64, 2, strides=2)
        # Deconvolution, image shape: (batch, 96, 96, 3)
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 32, 5, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 64, 5, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 128, 3, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.layers.conv2d(x, 256, 3, padding="same")
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2, padding="same")
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from real image input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.0001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.0001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())

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

            # Generate noise to feed to the generator
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

            # Prepare Targets (Real image: 1, Fake image: 0)
            # The first half of data fed to the discriminator are real images,
            # the other half are fake images (coming from the generator).
            batch_disc_y = np.concatenate(
                [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
            # Generator tries to fool the discriminator, thus targets are 1.
            batch_gen_y = np.ones([batch_size])

            # Training
            feed_dict = {real_image_input: batch_x, noise_input: z,
                         disc_target: batch_disc_y, gen_target: batch_gen_y}
            _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                    feed_dict=feed_dict)
            print('epoch: %d, step: %d, Generator Loss: %f, Discriminator Loss: %f' % (epoch, step, gl, dl))

        # Generate images from noise, using the generator network.
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(10):
            # Noise input.
            z = np.random.uniform(-1., 1., size=[4, noise_dim])
            g = sess.run(gen_sample, feed_dict={noise_input: z})
            for j in range(4):
                # Generate image from noise. Extend to 3 channels for matplot figure.
                img = g[j]
                a[j][i].imshow(img)
        f.savefig("faces_result/faces_%d.png" % epoch)
        plt.close()