import tensorflow as tf
import problem_unittests as tests
import helper
import pickle

x, y, keep_prob, cost, optimizer, accuracy = 0, 0, 0, 0, 0, 0

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, shape=[None]+list(image_shape), name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, shape=[None, n_classes], name="labels")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name="keep_prob")


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides, name="conv"):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    with tf.name_scope(name):
        # Create weight and bias
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal(list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.1))
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.constant(0.1, shape=[conv_num_outputs]))
            variable_summaries(b)
        # Apply convolution and add bias
        with tf.name_scope('Wx_plus_b'):
            conv = tf.nn.conv2d(x_tensor, W, strides=[1] + list(conv_strides) + [1], padding='SAME') + b
            tf.summary.histogram('pre_activations', conv)
        # Apply ReLu activation function
        conv = tf.nn.relu(conv, name='activation')
        tf.summary.histogram('activations', conv)

        # Apply max pooling
        pool = tf.nn.max_pool(conv, ksize=[1] + list(pool_ksize) + [1], strides= [1] + list(pool_strides) + [1], padding='SAME')
        return pool


def conv2d(x_tensor, conv_num_outputs, conv_ksize, conv_strides, name = "conv"):
    """Apply convolution to x_tensor
    @param x_tensor: Tensorflow Tensor 
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    return tensor after convolution
    """
    with tf.name_scope(name):
        # Create weight and bias
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal(list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.1))
            variable_summaries(W)
        with tf.name_scope("biases"):
            b = tf.Variable(tf.constant(0.1, shape=[conv_num_outputs]))
            variable_summaries(b)

        # Apply convolution and add bias
        with tf.name_scope('Wx_plus_b'):
            conv = tf.nn.conv2d(x_tensor, W, strides=[1] + list(conv_strides) + [1], padding='SAME') + b
            tf.summary.histogram('pre_activations', conv)
        

        return conv

def pool2d(x_tensor, pool_ksize, pool_strides, name = "pool"):
    """
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    return tensor after pool 
    """
    with tf.name_scope(name):
        return tf.nn.max_pool(x_tensor,
                ksize = [1] + list(pool_ksize) + [1],
                strides = [1] + list(pool_strides) + [1], 
                padding = 'SAME') 

def avg_pool2d(x_tensor, pool_ksize, pool_strides, name = "pool"):
    """
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    return tensor after pool 
    """
    with tf.name_scope(name):
        return tf.nn.avg_pool(x_tensor,
                ksize = [1] + list(pool_ksize) + [1],
                strides = [1] + list(pool_strides) + [1], 
                padding = 'SAME') 


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    with tf.name_scope('input_reshape'):
        x = x_tensor.get_shape().as_list()[1]
        y = x_tensor.get_shape().as_list()[2]
        z = x_tensor.get_shape().as_list()[3]
        image_shaped_input = tf.reshape(x_tensor, [-1, x*y*z])
        return image_shaped_input

def fully_conn(x_tensor, num_outputs, name="fc"):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    with tf.name_scope(name):
        return tf.layers.dense(x_tensor, num_outputs)

def output(x_tensor, num_outputs, name="output"):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    with tf.name_scope(name):
        return tf.layers.dense(x_tensor, num_outputs)


def resNet_block(x_tensor, bottleneck_d, num_outputs, _strides = (1, 1), short_cut = False, name = "resNet_block"):

    with tf.name_scope(name):
        shortcut = x_tensor

        """bottleneck desgin: 1x1 3x3 1x1 conv"""
        x_tensor = conv2d(x_tensor, bottleneck_d, (1, 1), (1, 1))
        x_tensor = tf.layers.batch_normalization(x_tensor) 
        x_tensor = tf.nn.relu(x_tensor)
        
        x_tensor = conv2d(x_tensor, bottleneck_d, (3, 3), _strides) 
        x_tensor = tf.layers.batch_normalization(x_tensor) 
        x_tensor = tf.nn.relu(x_tensor)

        x_tensor = conv2d(x_tensor, num_outputs, (1, 1), (1, 1))
        x_tensor = tf.layers.batch_normalization(x_tensor)

        if short_cut or _strides != (1, 1):
            shortcut = conv2d(shortcut, num_outputs, (1, 1), _strides)
            x_tensor = tf.layers.batch_normalization(x_tensor)
        
        # Identity
        x_tensor =  tf.add(x_tensor, shortcut)
        
        x_tensor = tf.nn.relu(x_tensor)
        
        print (x_tensor)
        return x_tensor 

# https://arxiv.org/pdf/1512.03385.pdf
def resNet(image, resNet_block):
    tf.summary.image('input', image)

    # Conv1
    with tf.variable_scope("conv1"):
        image = conv2d(image, 16, (3, 3), (1, 1))
        image = tf.layers.batch_normalization(image)
        image = tf.nn.relu(image)
    
    # Conv2
    for i in range (9):
        with tf.variable_scope("conv2_%d" % (i + 1)):
            if i == 0:
                # image = tf.nn.max_pool(image, ksize=[1, 3, 3, 1], strides= [1, 2, 2, 1], padding='SAME')
                image = resNet_block(image, 16, 64, short_cut = True)
            else:
                image = resNet_block(image, 16, 64)

    # Conv3
    for i in range(9):
        with tf.variable_scope("conv3_%d" % (i + 1)):
            if i == 0:
                image = resNet_block(image, 32, 128, _strides = (2, 2))
            else:
                image = resNet_block(image, 32, 128)
    
    # Conv4
    for i in range(9):
        with tf.variable_scope("conv4_%d" % (i + 1)):
            if i == 0:
                image = resNet_block(image, 64, 256, _strides = (2, 2))
            else:
                image = resNet_block(image, 64, 256)
    """
    # Conv5
    for i in range(3):
        with tf.variable_scope("conv5_%d" % (i + 1)):
            if i == 0:
                image = resNet_block(image, 128, 256, _strides = (2, 2))
            else:
                image = resNet_block(image, 128, 256)
    """
    
    # Avg Pool
    # image = tf.layers.batch_normalization(image)
    # image = tf.nn.relu(image)
    image = avg_pool2d(image, (8, 8), (1, 1))
    
    # Reshape
    image = flatten(image)

    # FC
    image = fully_conn(image, 10)

    print (image)
    return image

def vgg_net(x, keep_prob):
    
    tf.summary.image('input', x)
    x = conv2d(x, 16, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = pool2d(x, (2, 2), (2, 2))
    x= tf.nn.dropout(x, keep_prob, name="dropout1")

    x = conv2d(x, 32, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = pool2d(x, (2, 2), (2, 2))
    x = tf.nn.dropout(x, keep_prob, name="dropout1")

    x = conv2d(x, 64, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x= tf.nn.dropout(x, keep_prob, name="dropout1")

    x = conv2d(x, 64, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x= tf.nn.dropout(x, keep_prob, name="dropout1")

    x = conv2d(x, 128, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob, name="dropout1")

    x = conv2d(x, 128, (3, 3), (1, 1))
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.dropout(x, keep_prob, name="dropout1")


    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flatten1 = flatten(dropout1)
    
    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(flatten1, 256, "fc1")

    dropout2 = tf.nn.dropout(fc1, keep_prob, name="dropout2")

    fc2 = fully_conn(dropout2, 256, "fc2")

    dropout3 = tf.nn.dropout(fc2, keep_prob, name="dropout3")
    
    # Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    return output(dropout3, 10)



def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    tf.summary.image('input', x)

    convpool1 = conv2d_maxpool(x, 16, (3, 3), (1, 1), (3, 3), (2, 2), "conv1")
    
    convpool2 = conv2d_maxpool(convpool1, 64, (5, 5), (1, 1), (3, 3), (2, 2), "conv2")
    
    dropout1 = tf.nn.dropout(convpool2, keep_prob, name="dropout1")
    
    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flatten1 = flatten(dropout1)
    
    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(flatten1, 256, "fc1")

    dropout2 = tf.nn.dropout(fc1, keep_prob, name="dropout2")

    fc2 = fully_conn(dropout2, 256, "fc2")

    dropout3 = tf.nn.dropout(fc2, keep_prob, name="dropout3")
    
    # Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    return output(dropout3, 10)

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch, writer, merged_summary, i):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    if i % 5 == 0:
        s = session.run(merged_summary, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})
        writer.add_summary(s, i)
    session.run(optimizer, feed_dict={x:feature_batch, y:label_batch, keep_prob:keep_probability})

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # Load the Preprocessed Validation data
    valid_features, valid_labels = pickle.load(open(helper.pickle_file_path('preprocess_validation.p'), mode='rb'))

    loss = session.run(cost, feed_dict= {x:feature_batch, y:label_batch, keep_prob:1.0})
    valid_acc = session.run(accuracy, feed_dict= {x:valid_features, y:valid_labels, keep_prob: 1.0})
    print(loss)
    print(valid_acc)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def test_implementation():
    tf.reset_default_graph()
    tests.test_nn_image_inputs(neural_net_image_input)
    tests.test_nn_label_inputs(neural_net_label_input)
    tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)
    tests.test_con_pool(conv2d_maxpool)
    tests.test_flatten(flatten)
    tests.test_fully_conn(fully_conn)
    tests.test_output(output)

    build_cnn()

    tests.test_conv_net(conv_net)
    tests.test_train_nn(train_neural_network)

def build_cnn():
    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    global x, y, keep_prob
    x = neural_net_image_input((32, 32, 3))
    y = neural_net_label_input(10)
    keep_prob = neural_net_keep_prob_input()

    # Model
    logits = resNet(x, resNet_block)
    
    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss
    with tf.name_scope("xent"):
        global cost
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    tf.summary.scalar("cross_entropy", cost)

    # Optimizer
    with tf.name_scope("train"):
        global optimizer
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        with tf.name_scope("accuracy"):
            global accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)

def train_cnn_single_batch(epochs, batch_size, keep_probability):
    print('Checking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

def train_cnn_all_batches(epochs, batch_size, keep_probability):
    save_model_path = '../model/image_classification'

    print('Training...')
    sess = tf.InteractiveSession()

    # Visualize graph and merge all the summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('../tmp/cifar/15' + '/train', sess.graph)
    test_writer = tf.summary.FileWriter('../tmp/cifar/15' + '/test')

    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    i = 0
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels, train_writer, merged, i)
                i += 1
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)

    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)

    train_writer.close()
    test_writer.close()

# test_implementation()
build_cnn()
#train_cnn_single_batch(10, 256, 0.5)
train_cnn_all_batches(20, 256, 0.5)
