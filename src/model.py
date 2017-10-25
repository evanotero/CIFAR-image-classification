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
    return tf.placeholder(tf.float32, shape=[None, n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(tf.float32, name="keep_prob")


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
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
    # Create weight and bias
    W = tf.Variable(tf.truncated_normal(list(conv_ksize) + [x_tensor.get_shape().as_list()[3], conv_num_outputs], stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[conv_num_outputs]))

    # Apply convolution and add bias
    conv = tf.nn.conv2d(x_tensor, W, strides=[1] + list(conv_strides) + [1], padding='SAME') + b

    # Apply ReLu activation function
    conv = tf.nn.relu(conv)

    # Apply max pooling
    pool = tf.nn.max_pool(conv, ksize=[1] + list(pool_ksize) + [1], strides= [1] + list(pool_strides) + [1], padding='SAME')

    return pool

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    x = x_tensor.get_shape().as_list()[1]
    y = x_tensor.get_shape().as_list()[2]
    z = x_tensor.get_shape().as_list()[3]
    return tf.reshape(x_tensor, [-1, x*y*z])

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.layers.dense(x_tensor, num_outputs)

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    return tf.layers.dense(x_tensor, num_outputs)

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
    convpool1 = conv2d_maxpool(x, 8, (3,3), (1, 1), (3, 3), (2, 2))
    
    convpool2 = conv2d_maxpool(convpool1, 64, (2, 2), (1, 1), (3, 3), (2, 2))
    
    dropout2 = tf.nn.dropout(convpool2, keep_prob)
    
    # Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flatten1 = flatten(dropout2)
    
    # Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    fc1 = fully_conn(flatten1, 256)

    dropout3 = tf.nn.dropout(fc1, keep_prob)

    fc2 = fully_conn(dropout3, 256)

    dropout4 = tf.nn.dropout(fc2, keep_prob)
    
    # Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    out= output(dropout4, 10)

    # return output
    return out

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    """
    for batch, label in zip (feature_batch, label_batch):
        x = neural_net_image_input(batch.shape)
        y = neural_net_label_input(label.shape[0])
        logits = conv_net(x, keep_prob)
        logits = tf.identity(logits, name = 'logits') 
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y))
    
        
        session.run(optimizer.minimize(cost))
    """
    session.run(optimizer, feed_dict= {x:feature_batch, y:label_batch, keep_prob:keep_probability})

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
    logits = conv_net(x, keep_prob)

    # Name logits Tensor, so that is can be loaded from disk after training
    logits = tf.identity(logits, name='logits')

    # Loss and Optimizer
    global cost, optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Accuracy
    global accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

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
    save_model_path = '../image_classification'

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
                
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)

# test_implementation()
build_cnn()
# train_cnn_single_batch(10, 256, 0.5)
train_cnn_all_batches(10, 256, 0.5)
