#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import os
from tensorflow.python.lib.io import file_io


def _write_assets(assets_directory, assets_filename):
    if not file_io.file_exists(assets_directory):
        file_io.recursive_create_dir(assets_directory)

    path = os.path.join(
        tf.compat.as_bytes(assets_directory), tf.compat.as_bytes(assets_filename)
    )
    file_io.write_string_to_file(path, "asset-file-contents")
    return path


def _build_regression_signature(input_tensor, output_tensor):
    """Helper function for building a regression SignatureDef."""
    input_tensor_info = tf.saved_model.utils.build_tensor_info(input_tensor)
    signature_inputs = {
        tf.saved_model.signature_constants.REGRESS_INPUTS: input_tensor_info
    }
    output_tensor_info = tf.saved_model.utils.build_tensor_info(output_tensor)
    signature_outputs = {
        tf.saved_model.signature_constants.REGRESS_OUTPUTS: output_tensor_info
    }
    return tf.saved_model.signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        tf.saved_model.signature_constants.REGRESS_METHOD_NAME)


def predict():
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './tf_dir/')
        sess.run(tf.global_variables_initializer())

        input_x = sess.graph.get_tensor_by_name('x:0')
        # input_y = sess.graph.get_tensor_by_name('y:0')

        y_pred = sess.graph.get_tensor_by_name('y_pred:0')

        result = sess.run(y_pred, feed_dict={input_x: [[5]]})
        print(result)


def main():
    builder = tf.saved_model.builder.SavedModelBuilder('./tf_dir/')

    # Generate input data
    n_samples = 1000
    sample = 1000
    learning_rate = 0.01
    # batch_size = 100
    n_steps = 10000

    x_data = np.arange(100, step=.1)
    # y_data = x_data + 20 * np.sin(x_data / 10)
    y_data = 20 * x_data + 30

    x_data = np.reshape(x_data, (n_samples, 1))
    y_data = np.reshape(y_data, (n_samples, 1))

    # # Placeholders for batched input
    # x = tf.placeholder(tf.float32, shape=1, name='x')
    # y = tf.placeholder(tf.float32, shape=1, name='y')

    with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(log_device_placement=True)) as sess:
        # w = tf.get_variable('weights', 1, initializer=tf.random_normal_initializer())
        # b = tf.get_variable('bias', 1, initializer=tf.constant_initializer(0))

        w = tf.Variable(3.0, name='w')

        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')

        feature_configs = {
            'x': tf.FixedLenFeature([1], dtype=tf.float32),
            'y': tf.FixedLenFeature([1], dtype=tf.float32)
        }

        tf_example = tf.parse_example(serialized_tf_example, feature_configs)

        # use tf.identity() to assign name
        x = tf.identity(tf_example['x'], name='x')
        y = tf.identity(tf_example['y'], name='y')
        y_pred = tf.add(tf.multiply(x, w), y, name='y_pred')

        # y_pred = tf.matmul(x, w) + b
        # y_pred = x * w + b
        # loss = tf.reduce_sum((y - y_pred) ** 2 / n_samples)

        # opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        original_assets_directory = '/Users/zhangjin13/Desktop/assets'
        original_assets_filename = 'foo.txt'
        original_assets_filepath = _write_assets(original_assets_directory, original_assets_filename)

        assets_filepath = tf.constant(original_assets_filepath)
        tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
        filename_tensor = tf.Variable(
            original_assets_filename,
            name='filename_tensor',
            trainable=False,
            collections=[]
        )
        assign_filename_op = filename_tensor.assign(original_assets_filename)

        predict_input_tensor_x = tf.saved_model.utils.build_tensor_info(x)
        predict_input_tensor_y = tf.saved_model.utils.build_tensor_info(y)
        prediction_signature_inputs = {'x': predict_input_tensor_x, 'y': predict_input_tensor_y}

        predict_output_tensor = tf.saved_model.utils.build_tensor_info(y_pred)
        prediction_signature_outputs = {'y_pred': predict_output_tensor}

        prediction_signature_def = (
            tf.saved_model.signature_def_utils.build_signature_def(
                prediction_signature_inputs, prediction_signature_outputs,
                tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )

        signature_def_map = {
            'regress_x_to_y':
                _build_regression_signature(serialized_tf_example, y_pred),
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature_def
        }

        sess.run(tf.global_variables_initializer())

        # for _ in range(n_steps):
        #     index = np.random.randint(0, n_samples)
        #     x_ = x_data[index]
        #     y_ = y_data[index]
        #     _, loss_val = sess.run([opt, loss], feed_dict={x: [x_], y: [y_]})
        #     print(w.eval())
        #     print(b.eval())
        #     print(loss_val)

        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map=signature_def_map,
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
            main_op=tf.group(tf.saved_model.main_op.main_op(),
                             assign_filename_op)
        )

        builder.save()


if __name__ == '__main__':
    # main()
    predict()

    # inputs = {
    #     'input_x': tf.saved_model.utils.build_tensor_info(x),
    #     'input_y': tf.saved_model.utils.build_tensor_info(y)
    # }
    #
    # outputs = {'output': tf.saved_model.utils.build_tensor_info(loss)}
    # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
    #     inputs=inputs,
    #     outputs=outputs,
    #     method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    # )
    #
    # builder.add_meta_graph_and_variables(
    #     sess,
    #     [tf.saved_model.tag_constants.SERVING],
    #     {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}
    # )
    #
    # builder.save()
