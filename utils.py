import pickle # import_data, write_measures
import numpy as np # shuffle_np_arrays, get_2d_input
import tensorflow as tf # accuracy_vector_targets, get_fnn_model_name
import os # create_dir
import config


def shuffle_np_arrays(x, y):
    '''
    This only shuffle two numpy arrays along 0-dimension.
    Reference: https://tech.pic-collage.com/tips-of-numpy-shuffle-multiple-arrays-e4fb3e7ae2a
    '''

    # The dimension to shuffle is 0.
    dim_to_shuffle = 0

    # Generate the permutation index array.
    permutation = np.random.permutation(x.shape[dim_to_shuffle])

    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_x = x[permutation]
    shuffled_y = y[permutation]

    return shuffled_x, shuffled_y


def get_batch(i_batch, batch_size, input_train, output_train):
    # i_bach should start with 0.
    j_start = i_batch * batch_size
    j_end = (i_batch + 1) * batch_size
    batch_input = input_train[j_start:j_end]
    batch_output = output_train[j_start:j_end]

    return batch_input, batch_output


def get_measures(targets, predictions):
    '''
    targets: true target vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    predictions: predicted vectors
     - shape: (examples, vector_dimension)
     - The elements of vectors are only 0 or 1.
    '''
    n_examples = tf.shape(targets)[0]
    n_dimensions = tf.shape(targets)[1]

    equal = tf.cast(tf.equal(targets, predictions), tf.int32)

    # Measure 1: (target) operation accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    tensor_op_correct = tf.equal(digits_correct, tf.ones_like(digits_correct) * n_dimensions)
    tensor_op_correct = tf.cast(tensor_op_correct, tf.int32)

    op_correct = tf.reduce_sum(tensor_op_correct)
    op_wrong = n_examples - op_correct
    op_accuracy = tf.cast(op_correct, tf.float64) / tf.cast(n_examples, tf.float64)

    # Measure 2: digits_mean_accuracy
    digits_correct = tf.reduce_sum(equal, axis=1)
    digits_wrong = (tf.ones_like(digits_correct) * n_dimensions) - digits_correct
    digits_mean_correct = tf.reduce_mean(tf.cast(digits_correct, tf.float64))
    digits_mean_wrong = tf.reduce_mean(tf.cast(digits_wrong, tf.float64))
    digits_mean_accuracy = digits_mean_correct / tf.cast(n_dimensions, tf.float64)

    # Measure 3: per_digit_accuracy
    per_digit_correct = tf.reduce_sum(equal, axis=0)
    per_digit_wrong = (tf.ones_like(per_digit_correct) * n_examples) - per_digit_correct
    per_digit_accuracy = per_digit_correct / n_examples

    return (op_accuracy, op_wrong, op_correct,
            digits_mean_accuracy, digits_mean_wrong, digits_mean_correct,
            per_digit_accuracy, per_digit_wrong, per_digit_correct)


def get_fnn_model_name(run_id, tfnn_hidden_activation, list_layer_dims, str_optimizer, float_learning_rate, int_batch_size, epoch, str_acc_set, accuracy):

    if tfnn_hidden_activation == tf.nn.sigmoid:
        str_hidden_activation = 'sigm'
    if tfnn_hidden_activation == tf.nn.tanh:
        str_hidden_activation = 'tanh'
    if tfnn_hidden_activation == tf.nn.relu:
        str_hidden_activation = 'relu'

    str_model_name = run_id + '-' + 'fnn' + '-' + str_hidden_activation
    for dim in list_layer_dims:
        str_model_name = str_model_name + '-' + str(dim)

    str_model_name = '%s-%s-lr%f-bs%d-epoch%d-%sacc%.3f'%(str_model_name, str_optimizer, float_learning_rate, int_batch_size, epoch, str_acc_set, accuracy)

    return str_model_name


def dec2bin_embed(integer, n_binary_digits):
    list_embedding = list()
    str_binary = bin(integer)[2:]
    n_leading_zeros = n_binary_digits - len(str_binary)
    str_embedding = str_binary
    if n_leading_zeros > 0:
        str_embedding = '0' * n_leading_zeros + str_binary
    for str_digit in str_embedding:
        list_embedding.append(int(str_digit))
    return list_embedding


def dec2bin_np_embed(integer, n_binary_digits):
    return np.asarray(dec2bin_embed(integer, n_binary_digits), dtype=np.float).reshape(1,n_binary_digits)


def get_fcn_input(decimal_n1, decimal_n2, n_binary_digits):
    '''
    Get the binary input vector of addition of decimal_n1 and decimal_n2
    Parameters
    - decimal_n1, decimal_n1: int in the range of [0,2**(n_binary_digits)-1]
    Return
    - np.ndarry
    - shape = (1, 2 * n_binary_digits)
    - dtype = np.float
    '''
    binary_n1 = dec2bin_embed(decimal_n1, n_binary_digits)
    binary_n2 = dec2bin_embed(decimal_n2, n_binary_digits)
    input_vector = np.asarray(binary_n1 + binary_n2, dtype=np.float).reshape(1,2 * n_binary_digits)

    return input_vector


def get_fcn_target(decimal_n1, decimal_n2, n_binary_digits):
    '''
    Get the binary target vector of addition of decimal_n1 and decimal_n2
    Parameters
    - decimal_n1, decimal_n1: int in the range of [0,2**(n_binary_digits)-1]
    Return
    - np.ndarry
    - shape = (1, n_binary_digits + 1)
    - dtype = np.float
    '''
    decimal_target = decimal_n1 + decimal_n2
    binary_target = dec2bin_embed(decimal_target, n_binary_digits + 1)
    target_vector = np.asarray(binary_target, dtype=np.float).reshape(1, n_binary_digits + 1)

    return target_vector


def decode_fcn_output(np_output):
    '''
    Parameter
    - np_output: numpy.ndarray with shape=(1,n_output_binary_digits)
    '''
    n_output_binary_digits = np_output.shape[1]
    binary_string = ''
    for i in range(n_output_binary_digits):
        binary_string = binary_string + str(int(np_output[0,i]))

    return int(binary_string, base=2)


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def tf_tlu(x, name=None):
    return tf.cast(tf.greater(x, tf.ones_like(x) * 0.5), tf.float32, name=name)


def get_2d_inputs(inputs_1d):
    '''
    inputs_1d
      - shape: (n_examples, dim_input_data_1d)
      - dim_input_data_1d should be even.
    Return: inputs_2d
      - shape: (n_examples, 2, dim_input_data_1d // 2)
    '''
    inputs_2d = np.reshape(inputs_1d, (inputs_1d.shape[0], 2, -1, 1))

    return inputs_2d


def get_seq_data(inputs, targets):
    '''
    Parameters
    -------
    inputs : numpy.ndarray
           shape: (n_input, dim_input)

    outputs : numpy.ndarray
            shape : (n_input, dim_output)
    '''

    n_seq = inputs.shape[0]

    # Inputs
    dim_input = inputs.shape[1]
    n_operand = 2
    dim_operand = dim_input // n_operand

    tmp_inputs = np.reshape(inputs, (n_seq, n_operand, dim_operand))
    tmp_inputs = np.insert(tmp_inputs, 0, 0, axis=2) # Add 0 to the highest digit.
    tmp_inputs = np.flip(tmp_inputs, axis=2) # Axes 0 and 1 do not get flipped.
    seq_inputs = np.transpose(tmp_inputs, axes=(0, 2, 1)) # dim 2 to 1, dim 1 to 2

    # Outputs
    dim_target = targets.shape[1]
    seq_targets = np.reshape(np.flip(targets, axis=1), (n_seq, dim_target, 1))

    return (seq_inputs, seq_targets)


def get_str_activation(tf_activation):
    if tf_activation == tf.nn.sigmoid:
        str_activation = 'sigmoid'
    if tf_activation == tf.nn.tanh:
        str_activation = 'tanh'
    if tf_activation == tf.nn.relu:
        str_activation = 'relu'

    return str_activation


def init_run_info(NN_OUTPUT_DIM):
    # Initialize an empty dictionary
    run_info = dict()

    # Training info
    run_info['last_test_loss'] = -1
    run_info['last_test_accuracy'] = -1
    run_info['last_test_op_wrong'] = -1
    run_info['last_tlu_test_loss'] = -1
    run_info['last_tlu_test_accuracy'] = -1
    run_info['last_tlu_op_wrong'] = -1
    for i in range(NN_OUTPUT_DIM):
        run_info['last_digit-{}_accuracy'.format(i+1)] = -1
        run_info['last_digit-{}_wrong'.format(i+1)] = -1

    ## float epochs
    run_info['last_epoch'] = -1
    run_info['init_all_correct_epoch'] = -1
    for i in range(NN_OUTPUT_DIM):
        run_info['init_all_correct_digit-{}_epoch'.format(i+1)] = -1
        run_info['init_complete_all_correct_digit-{}_epoch'.format(i+1)] = -1

    return run_info


def write_run_info(run_info, float_epoch,
                   dev_run_outputs, dev_tlu_run_outputs, carry_run_outputs=None):

    (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
     per_digit_accuracy_val, per_digit_wrong_val) = dev_run_outputs

    if dev_tlu_run_outputs != None:
        (dev_loss_tlu_val, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = dev_tlu_run_outputs

    experiment_name = run_info['experiment_name']
    run_id = run_info['run_id']

    # loss, accuracy, n_wrong
    run_info['last_test_loss'] = dev_loss_val
    run_info['last_test_accuracy'] = dev_accuracy_val
    run_info['last_test_op_wrong'] = dev_op_wrong_val
    if dev_tlu_run_outputs != None:
        run_info['last_tlu_test_loss'] = dev_loss_tlu_val
        run_info['last_tlu_test_accuracy'] = dev_accuracy_tlu_val
        run_info['last_tlu_op_wrong'] = dev_op_wrong_tlu_val
    for i in range(len(per_digit_wrong_val)):
        run_info['last_digit-{}_accuracy'.format(i+1)] = per_digit_accuracy_val[-(i+1)]
        run_info['last_digit-{}_wrong'.format(i+1)] = per_digit_wrong_val[-(i+1)]
    if carry_run_outputs != None:
        for n_carries in carry_run_outputs.keys():
            carry_accuracy_val = carry_run_outputs[n_carries][1]
            carry_op_wrong_val = carry_run_outputs[n_carries][2]
            run_info['last_carry-{}_accuracy'.format(n_carries)] = carry_accuracy_val
            run_info['last_carry-{}_wrong'.format(n_carries)] = carry_op_wrong_val

    # float epochs
    run_info['last_epoch'] = float_epoch

    # The float epoch of all correct operation float epoch
    if dev_op_wrong_val == 0 and run_info['init_all_correct_epoch'] == -1:
        run_info['init_all_correct_epoch'] = float_epoch

    # The float epoch of all correct digit
    for i in range(len(per_digit_wrong_val)):
        # init_all_correct: the initial time to attain all correct digit outputs.
        init_all_correct_key = 'init_all_correct_digit-{}_epoch'.format(i+1)
        # init_complete_all_correct: the last initial time to attain all correct digit outputs.
        init_complete_all_correct_key = 'init_complete_all_correct_digit-{}_epoch'.format(i+1)

        if per_digit_wrong_val[-(i+1)] == 0 and run_info[init_all_correct_key] == -1:
            run_info[init_all_correct_key] = float_epoch
        if per_digit_wrong_val[-(i+1)] == 0 and run_info[init_complete_all_correct_key] == -1:
            run_info[init_complete_all_correct_key] = float_epoch
        if per_digit_wrong_val[-(i+1)] != 0 and run_info[init_complete_all_correct_key] != -1:
            run_info[init_complete_all_correct_key] = -1

    # The float epoch of all carry datasets
    if carry_run_outputs != None:
        for n_carries in carry_run_outputs.keys():
            carry_op_wrong_val = carry_run_outputs[n_carries][2]
            # init_all_correct: the initial time to attain all correct output for `n_carries` dataset.
            init_all_correct_key = 'init_all_correct_carry-{}_epoch'.format(n_carries)
            # init_complete_all_correct: the last initial time to attain all correct output for `n_carries` dataset.
            init_complete_all_correct_key = 'init_complete_all_correct_carry-{}_epoch'.format(n_carries)

            # Initialization step
            if init_all_correct_key not in run_info:
                run_info[init_all_correct_key] = -1
            if init_complete_all_correct_key not in run_info:
                run_info[init_complete_all_correct_key] = -1

            if carry_op_wrong_val == 0 and run_info[init_all_correct_key] == -1:
                run_info[init_all_correct_key] = float_epoch
            if carry_op_wrong_val == 0 and run_info[init_complete_all_correct_key] == -1:
                run_info[init_complete_all_correct_key] = float_epoch
            if carry_op_wrong_val != 0 and run_info[init_complete_all_correct_key] != -1:
                run_info[init_complete_all_correct_key] = -1

    # Save run_info
    create_dir('{}/{}'.format(config.dir_run_info_experiments(), experiment_name))
    with open('{}/{}/run-{}.pickle'.format(config.dir_run_info_experiments(), experiment_name, run_id), 'wb') as f:
        pickle.dump(run_info, f)


def write_measures(run_info, float_epoch,
                   dev_run_outputs, dev_tlu_run_outputs):

    (dev_loss_val, dev_accuracy_val, dev_op_wrong_val,
     per_digit_accuracy_val, per_digit_wrong_val) = dev_run_outputs

    if dev_tlu_run_outputs != None:
        (dev_loss_tlu_val, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = dev_tlu_run_outputs

    run_id = run_info['run_id']
    experiment_name = run_info['experiment_name']

    create_dir('{}/{}'.format(config.dir_measure_log(), experiment_name))
    pickle_path = '{}/{}/run-{}.pickle'.format(config.dir_measure_log(), experiment_name, run_id)

    if not os.path.exists(pickle_path):
        # Create a new measure log dictionary
        measure_logs = dict()
        measure_logs['float_epoch'] = list()
        measure_logs['test_loss'] = list()
        measure_logs['test_accuracy'] = list()
        measure_logs['test_op_wrong'] = list()
        if dev_tlu_run_outputs != None:
            measure_logs['tlu_test_loss'] = list()
            measure_logs['tlu_test_accuracy'] = list()
            measure_logs['tlu_op_wrong'] = list()

        for i in range(len(per_digit_wrong_val)):
            measure_logs['digit-{}_accuracy'.format(i+1)] = list()
            measure_logs['digit-{}_op_wrong'.format(i+1)] = list()

    else:
        # Import the measure log dictionary from the pickle file.
        with open(pickle_path, 'rb') as f:
            measure_logs = pickle.load(f)

    # Append a new set of measures
    measure_logs['float_epoch'].append(float_epoch)
    measure_logs['test_loss'].append(dev_loss_val)
    measure_logs['test_accuracy'].append(dev_accuracy_val)
    measure_logs['test_op_wrong'].append(dev_op_wrong_val)
    if dev_tlu_run_outputs != None:
        measure_logs['tlu_test_loss'].append(dev_loss_tlu_val)
        measure_logs['tlu_test_accuracy'].append(dev_accuracy_tlu_val)
        measure_logs['tlu_op_wrong'].append(dev_op_wrong_tlu_val)

    for i in range(len(per_digit_wrong_val)):
        measure_logs['digit-{}_accuracy'.format(i+1)].append(per_digit_accuracy_val[-(i+1)])
        measure_logs['digit-{}_op_wrong'.format(i+1)].append(per_digit_wrong_val[-(i+1)])

    # Write the appended measure_logs
    with open(pickle_path, 'wb') as f:
        pickle.dump(measure_logs, f)


def read_measure_logs(experiment_name, run_id):
    pickle_path = '{}/{}/run-{}.pickle'.format(config.dir_measure_log(), experiment_name, run_id)
    with open(pickle_path, 'rb') as f:
        measure_logs = pickle.load(f)

    return measure_logs
