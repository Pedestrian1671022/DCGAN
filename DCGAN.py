import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_pixels = 28
batch_size = 20
epochs = 400
train_size = 16412
train_tfrecord = "faces_train.tfrecord"

def read_and_decode(serialized_example):
    features = tf.parse_single_example(serialized_example, features={"image":tf.FixedLenFeature([], tf.string), "filename":tf.compat.v1.FixedLenFeature([], tf.compat.v1.string)})
    img = tf.decode_raw(features["image"], tf.uint8)
    img = tf.reshape(img, [image_pixels, image_pixels, 3])

    filename = tf.compat.v1.cast(features["filename"], tf.compat.v1.string)
    return img, filename

def conv_batch_normalization(prev_layer, layer_depth, is_training):
    gamma = tf.Variable(tf.ones([layer_depth]), trainable=True)
    beta = tf.Variable(tf.zeros([layer_depth]), trainable=True)
    pop_mean = tf.Variable(tf.zeros([layer_depth]), trainable=False)
    pop_variance = tf.Variable(tf.ones([layer_depth]), trainable=False)
    epsilon = 1e-3
    def batch_norm_training():
        batch_mean, batch_variance = tf.nn.moments(prev_layer, [0, 1, 2], keep_dims=False)
        decay = 0.99
        train_mean = tf.assign(pop_mean, pop_mean*decay + batch_mean*(1 - decay))
        train_variance = tf.assign(pop_variance, pop_variance*decay + batch_variance*(1 - decay))
        with tf.control_dependencies([train_mean, train_variance]):
            return tf.nn.batch_normalization(prev_layer, batch_mean, batch_variance, beta, gamma, epsilon)
    def batch_norm_inference():
        return tf.nn.batch_normalization(prev_layer, pop_mean, pop_variance, beta, gamma, epsilon)
    batch_normalized_output = tf.cond(is_training, batch_norm_training, batch_norm_inference)
    return batch_normalized_output

with tf.name_scope("generator"):
    with tf.name_scope("gen_input"):
        noise = tf.placeholder(tf.float32, [None, 100], name="noise")
        is_training = tf.placeholder(tf.bool, None, name="is_training")
    with tf.name_scope("gen"):
        W1_gen = tf.Variable(tf.truncated_normal([100, 128*7*7], stddev=0.1), trainable=True, name="W1_gen")
        h1_gen = tf.nn.relu(tf.matmul(noise, W1_gen))
        h1_gen_reshaped = tf.reshape(h1_gen, [-1, 7, 7, 128])
        h1_gen_reshaped_upsampled = tf.nn.conv2d_transpose(h1_gen_reshaped, tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)), output_shape=[batch_size, 14, 14, 128], strides=[1, 2, 2, 1], padding="SAME")


        W2_gen = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1), trainable=True, name="W2_gen")
        b2_gen = tf.Variable(tf.constant(0.0, shape=[128]), trainable=True, name="b2_gen")
        h2_gen = tf.nn.bias_add(tf.nn.conv2d(h1_gen_reshaped_upsampled, W2_gen, strides=[1, 1, 1, 1], padding="SAME"), b2_gen)
        h2_gen_bn_relu = tf.nn.relu(conv_batch_normalization(prev_layer=h2_gen, layer_depth=128, is_training=is_training))
        h2_gen_reshaped_upsampled = tf.nn.conv2d_transpose(h2_gen_bn_relu, tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1)), output_shape=[batch_size, 28, 28, 128], strides=[1, 2, 2, 1], padding="SAME")


        W3_gen = tf.Variable(tf.truncated_normal([3, 3, 128, 64], stddev=0.1), trainable=True, name="W3_gen")
        b3_gen = tf.Variable(tf.constant(0.0, shape=[64]), trainable=True, name="b3_gen")
        h3_gen = tf.nn.bias_add(tf.nn.conv2d(h2_gen_reshaped_upsampled, W3_gen, strides=[1, 1, 1, 1], padding="SAME"), b3_gen)
        h3_gen_bn_relu = tf.nn.relu(conv_batch_normalization(prev_layer=h3_gen, layer_depth=64, is_training=is_training))
    with tf.name_scope("gen_output"):
        W4_gen = tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.1), trainable=True, name="W4_gen")
        b4_gen = tf.Variable(tf.constant(0.0, shape=[3]), trainable=True, name="b4_gen")
        h4_gen = tf.nn.bias_add(tf.nn.conv2d(h3_gen_bn_relu, W4_gen, strides=[1, 1, 1, 1], padding="SAME"), b4_gen)
        output_gen =  tf.nn.tanh(h4_gen)
        
with tf.variable_scope("discriminator_and_combined"):
    with tf.variable_scope("dis_input"):
        image = tf.placeholder(tf.float32, [None, 28, 28, 3], name="image")
        is_trainanble = tf.placeholder(tf.bool, None, name="is_trainable")
        rate = tf.placeholder(tf.float32, None, name="rate")

    with tf.variable_scope("dis_and_combined"):
        W1_dis_and_combined = tf.get_variable(shape=[3, 3, 3, 32], trainable=True, name="W1_dis_and_combined")
        b1_dis_and_combined = tf.get_variable(shape=[32], trainable=True, name="b1_dis_and_combined")

        h1_dis_leakyrelu = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(image, W1_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b1_dis_and_combined))
        h1_dis_leakyrelu_dropout = tf.nn.dropout(h1_dis_leakyrelu, rate=rate)

        h1_combined_leakyrelu = tf.nn.leaky_relu(tf.nn.bias_add(tf.nn.conv2d(output_gen, W1_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b1_dis_and_combined))
        h1_combined_leakyrelu_dropout = tf.nn.dropout(h1_combined_leakyrelu, rate=rate)


        W2_dis_and_combined = tf.get_variable(shape=[3, 3, 32, 64], trainable=True, name="W2_dis_and_combined")
        b2_dis_and_combined = tf.get_variable(shape=[64], trainable=True, name="b2_dis_and_combined")

        h2_dis = tf.nn.bias_add(tf.nn.conv2d(h1_dis_leakyrelu_dropout, W2_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b2_dis_and_combined)
        h2_dis_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h2_dis, layer_depth=64, is_training=is_trainanble))
        h2_dis_bn_leakyrelu_dropout = tf.nn.dropout(h2_dis_bn_leakyrelu, rate=rate)

        h2_combined = tf.nn.bias_add(tf.nn.conv2d(h1_combined_leakyrelu_dropout, W2_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b2_dis_and_combined)
        h2_combined_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h2_combined, layer_depth=64, is_training=is_trainanble))
        h2_combined_bn_leakyrelu_dropout = tf.nn.dropout(h2_combined_bn_leakyrelu, rate=rate)


        W3_dis_and_combined = tf.get_variable(shape=[3, 3, 64, 128], trainable=True, name="W3_dis_and_combined")
        b3_dis_and_combined = tf.get_variable(shape=[128], trainable=True, name="b3_dis_and_combined")

        h3_dis = tf.nn.bias_add(tf.nn.conv2d(h2_dis_bn_leakyrelu_dropout, W3_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b3_dis_and_combined)
        h3_dis_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h3_dis, layer_depth=128, is_training=is_trainanble))
        h3_dis_bn_leakyrelu_dropout = tf.nn.dropout(h3_dis_bn_leakyrelu, rate=rate)

        h3_combined = tf.nn.bias_add(tf.nn.conv2d(h2_combined_bn_leakyrelu_dropout, W3_dis_and_combined, strides=[1, 2, 2, 1], padding="SAME"), b3_dis_and_combined)
        h3_combined_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h3_combined, layer_depth=128, is_training=is_trainanble))
        h3_combined_bn_leakyrelu_dropout = tf.nn.dropout(h3_combined_bn_leakyrelu, rate=rate)


        W4_dis_and_combined = tf.get_variable(shape=[3, 3, 128, 256], trainable=True, name="W4_dis_and_combined")
        b4_dis_and_combined = tf.get_variable(shape=[256], trainable=True, name="b4_dis_and_combined")

        h4_dis = tf.nn.bias_add(tf.nn.conv2d(h3_dis_bn_leakyrelu_dropout, W4_dis_and_combined, strides=[1, 1, 1, 1], padding="SAME"), b4_dis_and_combined)
        h4_dis_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h4_dis, layer_depth=256, is_training=is_trainanble))
        h4_dis_bn_leakyrelu_dropout = tf.nn.dropout(h4_dis_bn_leakyrelu, rate=rate)

        h4_combined = tf.nn.bias_add(tf.nn.conv2d(h3_combined_bn_leakyrelu_dropout, W4_dis_and_combined, strides=[1, 1, 1, 1], padding="SAME"), b4_dis_and_combined)
        h4_combined_bn_leakyrelu = tf.nn.leaky_relu(conv_batch_normalization(prev_layer=h4_combined, layer_depth=256, is_training=is_trainanble))
        h4_combined_bn_leakyrelu_dropout = tf.nn.dropout(h4_combined_bn_leakyrelu, rate=rate)

    with tf.variable_scope("dis_and_combined_output"):
        W5_dis_and_combined = tf.get_variable(shape=[4 * 4 * 256, 1], trainable=True, name="W5_dis_and_combined")

        h5_dis = tf.reshape(h4_dis_bn_leakyrelu_dropout, [-1, 4*4*256])
        output_dis = tf.matmul(h5_dis, W5_dis_and_combined)

        h5_combined = tf.reshape(h4_combined_bn_leakyrelu_dropout, [-1, 4 * 4 * 256])
        output_combined = tf.matmul(h5_combined, W5_dis_and_combined)

with tf.name_scope("dis_and_combined_loss"):
    label_dis_and_combined = tf.placeholder(tf.float32, [None, 1], name="label_dis_and_combined")

    loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_dis_and_combined, logits=output_dis))

    loss_combined = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label_dis_and_combined, logits=output_combined))

with tf.name_scope("train"):
    lr = tf.Variable(initial_value=1e-4, trainable=False, name="learning_rate", dtype=tf.float32)
    update_learning_rate = tf.assign(lr, lr * 0.8)
    train_phase1 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss_dis, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator_and_combined"))
    train_phase2 = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss_combined, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator"))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    dataset_train = tf.data.TFRecordDataset([train_tfrecord])
    dataset_train = dataset_train.map(read_and_decode)
    dataset_train = dataset_train.repeat(epochs).shuffle(1000).batch(batch_size)
    iterator_train = dataset_train.make_initializable_iterator()
    next_element_train = iterator_train.get_next()
    sess.run(iterator_train.initializer)

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        if epoch > 100 and epoch % 10 == 0:
            sess.run(update_learning_rate)
        print("learning_rate:", sess.run(lr))
        for step in range(int(train_size/batch_size)):
            img_train, filename = sess.run(next_element_train)
            img_train = img_train/127.5-1
            # cv2.imshow("image", np.squeeze(img_train))
            # filename = filename[0].decode()
            # print(filename)
            # cv2.waitKey(0)
            _noise = np.random.normal(0, 1, (batch_size, 100))
            noise_image = sess.run(output_gen, feed_dict={noise: _noise, is_training:False})
            _, _loss_dis1 = sess.run([train_phase1, loss_dis], feed_dict={image: noise_image, label_dis_and_combined:fake, is_trainanble: True, rate:0.2})
            _, _loss_dis2 = sess.run([train_phase1, loss_dis], feed_dict={image: img_train, label_dis_and_combined:valid, is_trainanble: True, rate: 0.2})
            _, _loss_combined = sess.run([train_phase2, loss_combined], feed_dict={noise: _noise, label_dis_and_combined:valid, is_training:True, is_trainanble: False, rate: 0.0})
            if step % 10 == 0:
                print("epoch", epoch, " step:", step / 10, " loss_dis:", (_loss_dis1 + _loss_dis2)/2.0, " loss_combined:", _loss_combined)
        for i in range(batch_size):
            cv2.imwrite("images/" + str(epoch) + "_" + str(i) + ".jpg", np.squeeze(127.5 * noise_image[i, :, :, :] + 127.5))
        tf.train.Saver().save(sess, "ckpt/model.ckpt")
        print("save ckpt:", epoch)
    tf.summary.FileWriter("logs/", sess.graph)