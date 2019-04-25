import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import utils
import data_utils
from datetime import datetime
import os
import pickle
import sys
import config

def main():
    experiment_name = sys.argv[1]
    operand_bits =  int(sys.argv[2])
    operator =  sys.argv[3]
    hidden_units =  int(sys.argv[4])
    str_device_num = str(int(sys.argv[5]))
    nn_model_type = 'mlp'
    tlu_on = config.tlu_on()
    mlp_run(experiment_name, operand_bits, operator, hidden_units, str_device_num,
        nn_model_type, tlu_on)


def mlp_run(experiment_name, operand_bits, operator, hidden_units, str_device_num,
    nn_model_type, tlu_on):
    def train(sess, batch_input, batch_target, float_epoch, all_correct_val):
        _, _, _ = sess.run([loss, op_accuracy, train_op],
                            feed_dict={inputs:batch_input, targets:batch_target,
                                       condition_tlu:False,
                                       training_epoch:float_epoch,
                                       big_batch_training:big_batch_training_val,
                                       all_correct_epoch:(all_correct_val * float_epoch),
                                       all_correct:all_correct_val})

    def write_train_summary(sess, compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step):
        # Run computing train loss, accuracy
        train_loss, train_accuracy, merged_summary_op_val = sess.run(
            compute_nodes,
            feed_dict={inputs:batch_input, targets:batch_target,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("epoch: {}, step: {}, train_loss: {}, train_accuracy: {}".format(epoch, step, train_loss, train_accuracy))
        train_summary_writer.add_summary(merged_summary_op_val, step)

        return (train_loss, train_accuracy)

    def write_dev_summary(sess, compute_nodes, float_epoch, all_correct_val, step):

        dev_loss, dev_accuracy, merged_summary_op_val, dev_op_wrong_val, per_digit_accuracy_val, per_digit_wrong_val = sess.run(
            compute_nodes,
            feed_dict={inputs:input_dev, targets:target_dev,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("└ epoch: {}, step: {}, dev_loss: {}, dev_accuracy: {}, op_wrong: {}".format(epoch, step, dev_loss, dev_accuracy, op_wrong_val))
        dev_summary_writer.add_summary(merged_summary_op_val, step)

        return (dev_loss, dev_accuracy, dev_op_wrong_val, per_digit_accuracy_val, per_digit_wrong_val)

    def write_tlu_dev_summary(sess, compute_nodes, float_epoch, all_correct_val, step):
        dev_loss_tlu, dev_accuracy_tlu, merged_summary_op_val, dev_op_wrong_val_tlu, _, _ = sess.run(
            compute_nodes,
            feed_dict={inputs:input_dev, targets:target_dev,
                       condition_tlu:True,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})

        ##print("└ [TLU] epoch: {}, step: {}, dev_loss: {}, dev_accuracy: {}, op_wrong: {}".format(epoch, step, dev_loss_tlu, dev_accuracy_tlu, op_wrong_val_tlu))
        tlu_summary_writer.add_summary(merged_summary_op_val, step)

        return (dev_loss_tlu, dev_accuracy_tlu, dev_op_wrong_val_tlu)

    def write_test_summary(sess, compute_nodes, float_epoch, all_correct_val, step):
        test_loss, test_accuracy, merged_summary_op_val, op_wrong_val = sess.run(
            compute_nodes,
            feed_dict={inputs:input_test, targets:target_test,
                       condition_tlu:False,
                       training_epoch:float_epoch,
                       big_batch_training:big_batch_training_val,
                       all_correct_epoch:(all_correct_val * float_epoch),
                       all_correct:all_correct_val})
        print("└ epoch: {}, step: {}, test_loss: {}, test_accuracy: {}, op_wrong: {}".format(epoch, step, test_loss, test_accuracy, op_wrong_val))
        test_summary_writer.add_summary(merged_summary_op_val, step)

        return (test_loss, test_accuracy, op_wrong_val)

    def write_carry_datasets_summary(sess, compute_nodes, float_epoch, all_correct_val, step):
        value_dict = dict()
        for n_carries in carry_datasets.keys():
            carry_dataset_input = carry_datasets[n_carries]['input']
            carry_dataset_output = carry_datasets[n_carries]['output']

            carry_loss_val, carry_accuracy_val, merged_summary_op_val, carry_op_wrong_val, carry_per_digit_accuracy_val, carry_per_digit_wrong_val = sess.run(
                compute_nodes,
                feed_dict={inputs:carry_dataset_input, targets:carry_dataset_output,
                           condition_tlu:False,
                           training_epoch:float_epoch,
                           big_batch_training:big_batch_training_val,
                           all_correct_epoch:(all_correct_val * float_epoch),
                           all_correct:all_correct_val})

            value_dict[n_carries] = (carry_loss_val, carry_accuracy_val, carry_op_wrong_val, carry_per_digit_accuracy_val, carry_per_digit_wrong_val)
            carry_datasets_summary_writers[n_carries].add_summary(merged_summary_op_val, step)

        return value_dict

    def write_embeddings_summary(sess, h1):
        # Reference: https://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector
        dir_logs = os.path.join(config.dir_saved_models(), experiment_name)
        metadata = os.path.join(dir_logs, 'metadata.tsv')
        carry_datasets = data_utils.import_carry_datasets(operand_bits, operator)
        input_arrays = list()
        with open(metadata, 'w') as f:
            for carries in carry_datasets.keys():
                input_arrays.append(carry_datasets[carries]['input'])
                f.write('{}\n'.format(carries))

        carry_inputs = np.concatenate(input_arrays, axis=0)

        [h1_val] = sess.run([h1],
            feed_dict={inputs:carry_inputs,
                       condition_tlu:False})

        h1_var = tf.Variable(h1_val, name='h1_var')
        saver = tf.train.Saver([h1_var])
        sess.run(h1_var.initializer)
        saver.save(sess, os.path.join(dir_logs, 'h1_var.ckpt'))

        pconfig = projector.ProjectorConfig()
        pconfig.model_checkpoint_path = os.path.join(dir_logs, 'h1_var.ckpt')
        embedding = pconfig.embeddings.add()
        embedding.tensor_name = h1_var.name
        embedding.metadata_path = metadata
        projector.visualize_embeddings(tf.summary.FileWriter(dir_logs), pconfig)


    def create_carry_datasets_summary_writers(logdir, carry_datasets):
        carry_datasets_summary_writers = dict()
        for n_carries in carry_datasets.keys():
            carry_datasets_summary_writers[n_carries] =  tf.summary.FileWriter(logdir + '/carry-{}'.format(n_carries))
        return carry_datasets_summary_writers

    def close_carry_datasets_summary_writers(carry_datasets_summary_writers):
        for n_carries in carry_datasets_summary_writers.keys():
            carry_datasets_summary_writers[n_carries].close()

    def get_all_correct_val(op_wrong_val):
        if op_wrong_val == 0:
            return True
        else:
            return False

    def is_last_batch(i_batch):
        if i_batch == (n_batch - 1):
            return True
        else:
            return False

    def decrease_dev_summary_period(dev_accuracy_val, op_wrong_val):
        # Preconditions
        if not decreasing_dev_summary_period:
            return
        if dev_accuracy_val < 0.999:
            return

        # If the preconditions are satisfied, ...
        if op_wrong_val <= 8:
            dev_summary_period = int(init_dev_summary_period // 128)
        elif op_wrong_val <= 16:
            dev_summary_period = int(init_dev_summary_period // 64)
        if op_wrong_val <= 32:
            dev_summary_period = int(init_dev_summary_period // 32)
        elif op_wrong_val <= 64:
            dev_summary_period = int(init_dev_summary_period // 16)
        elif op_wrong_val <= 128:
            dev_summary_period = int(init_dev_summary_period // 8)

        if op_wrong_val > 512:
            dev_summary_period = init_dev_summary_period

    ############################################################################
    # Running point.

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]= str_device_num # 0, 1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Disable all debugging logs: Unable to display GPU info when running on the bash

    # Import datasets
    (train_ratio, dev_ratio, test_ratio) = config.dataset_ratio()
    (input_train, input_dev, input_test,
        target_train, target_dev, target_test
    ) = data_utils.import_op_dataset(operator, operand_bits,
            train_ratio=train_ratio, dev_ratio=dev_ratio, test_ratio=test_ratio)

    if operator in config.operators_list():
        carry_datasets = data_utils.import_carry_datasets(operand_bits, operator)

    # If the training dataset takes all examples, then the dev and test datasets are the same as the training one.
    if dev_ratio == 0.0 and test_ratio == 0.0:
        input_dev = input_train
        target_dev = target_train
        input_test = input_train
        target_test = target_train
    if dev_ratio == 0.0 and test_ratio != 0.0:
        input_dev = input_test
        target_dev = target_test

    # Contants
    NN_INPUT_DIM = input_train.shape[1]
    NN_OUTPUT_DIM = target_train.shape[1]

    # Hyperparameters - training
    batch_size = config.batch_size()
    big_batch_size = config.big_batch_size()
    n_epoch = config.n_epoch()
    learning_rate = config.learning_rate()
    all_correct_stop = config.all_correct_stop()
    big_batch_saturation = config.big_batch_saturation()
    if big_batch_saturation:
        all_correct_stop = False

    # Hyperparameters - model
    activation = config.activation() # tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu
    str_activation = utils.get_str_activation(activation)
    h_layer_dims = [hidden_units] # h_layer_dims[0]: dim of h1 layer
    last_size = NN_OUTPUT_DIM

    # Variables determined by other variables
    train_size = input_train.shape[0]
    n_batch = train_size // batch_size

    # Print periods
    train_summary_period = n_batch // 4 # 4 times per epoch
    init_dev_summary_period = n_batch # n_batch: print at every epoch
    dev_summary_period = init_dev_summary_period
    decreasing_dev_summary_period = config.decreasing_dev_summary_period()

    # Weight initialization
    ## https://www.tensorflow.org/api_docs/python/tf/contrib/layers/variance_scaling_initializer
    if activation == tf.nn.relu:
        init_factor = 2.0
    if activation == tf.nn.sigmoid:
        init_factor = 1.0
    if activation == tf.nn.tanh:
        init_factor = 1.0

    fan_in_1 = NN_INPUT_DIM
    fan_in_2 = h_layer_dims[0]

    ############################################################################
    # Creating a computational graph.

    # Initializing paraters to learn.
    with tf.name_scope('parameter'):
        W1 = tf.Variable(tf.truncated_normal((NN_INPUT_DIM, h_layer_dims[0]), stddev=np.sqrt(init_factor / fan_in_1)), name="W1")
        b1 = tf.Variable(tf.zeros((h_layer_dims[0])), name="b1")
        W2 = tf.Variable(tf.truncated_normal((h_layer_dims[0], NN_OUTPUT_DIM), stddev=np.sqrt(init_factor / fan_in_2)), name="W2")
        b2 = tf.Variable(tf.zeros((NN_OUTPUT_DIM)), name="b2")

    # Setting the input and target output.
    inputs = tf.placeholder(tf.float32, shape=(None, input_train.shape[1]), name='inputs') # None for mini-batch size
    targets = tf.placeholder(tf.float32, shape=(None, target_train.shape[1]), name='targets')

    condition_tlu = tf.placeholder(tf.int32, shape=(), name="tlu_condition")
    is_tlu_hidden = tf.greater(condition_tlu, tf.constant(0, tf.int32))
    #is_tlu_hidden = tf.constant(condition_tlu == True, dtype=tf.bool) # https://github.com/pkmital/tensorflow_tutorials/issues/36

    # NN structure
    with tf.name_scope('layer1'):
        h1_logits = tf.add(tf.matmul(inputs,  W1), b1)
        h1 = tf.cond(is_tlu_hidden, lambda: utils.tf_tlu(h1_logits, name='h1_tlu'), lambda: activation(h1_logits, name='h1')) # https://stackoverflow.com/questions/35833011/how-to-add-if-condition-in-a-tensorflow-graph / https://www.tensorflow.org/versions/r1.7/api_docs/python/tf/cond
    with tf.name_scope('layer2'):
        last_logits = tf.add(tf.matmul(h1,  W2), b2)
        sigmoid_outputs = tf.sigmoid(last_logits)
    predictions = utils.tf_tlu(sigmoid_outputs, name='predictions')

    # Loss: objective function
    with tf.name_scope('loss'):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=last_logits) # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        loss = tf.reduce_mean(loss)
        if config.l1_coef() != 0:
            loss = loss \
                + config.l1_coef() / (2 * batch_size) * (tf.reduce_sum(tf.abs(W1)) + tf.reduce_sum(tf.abs(W2)))
            #    + config.l1_coef() / (2 * batch_size) * (tf.reduce_sum(tf.abs(tf.abs(W1) - 1)) + tf.reduce_sum(tf.abs(tf.abs(W2) - 1)))
        if config.l2_coef() != 0:
            loss = loss \
                + config.l2_coef() / (2 * batch_size) * (tf.reduce_sum(tf.square(W1)) + tf.reduce_sum(tf.square(W2)))


    # Get measures:
    # [1] operation measures (accuracy, n_wrong, n_correct)
    # [2] mean digits accuracy (mean_digits_accuracy)
    # [3] per digit accuracy (per_digit_accuracy)
    (op_accuracy, op_wrong, op_correct,
     digits_mean_accuracy, digits_mean_wrong, digits_mean_correct,
     per_digit_accuracy, per_digit_wrong, per_digit_correct
    ) = utils.get_measures(targets, predictions)


    # Training, optimization
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    init = tf.global_variables_initializer()

    training_epoch = tf.placeholder(tf.float32, shape=None)
    all_correct_epoch = tf.placeholder(tf.float32, shape=None)
    big_batch_training = tf.placeholder(tf.int32, shape=None)
    all_correct = tf.placeholder(tf.int32, shape=None)

    # Summary: Scalar
    ## Measures
    tf.summary.scalar('loss', loss)

    with tf.name_scope('operation'):
        tf.summary.scalar('accuracy', op_accuracy)
        tf.summary.scalar('wrong', op_wrong)

    with tf.name_scope('digits'):
        tf.summary.scalar('mean_accuracy', digits_mean_accuracy)
        tf.summary.scalar('mean_wrong', digits_mean_wrong)

    with tf.name_scope('per_digit'):
        for i in range(NN_OUTPUT_DIM):
            tf.summary.scalar('digit-{}/accuracy'.format(i+1), per_digit_accuracy[-(i+1)])
            tf.summary.scalar('digit-{}/wrong'.format(i+1), per_digit_wrong[-(i+1)])
            # add per_digit_correct

    tf.summary.scalar('epoch', training_epoch)
    tf.summary.scalar('all_correct_epoch', all_correct_epoch)
    tf.summary.scalar('big_batch_training', big_batch_training)
    tf.summary.scalar('all_correct', all_correct)
    tf.summary.scalar('condition_tlu', condition_tlu)

    # Summary: Histogram
    with tf.name_scope('layer1'):
        tf.summary.histogram('weight', W1)
        tf.summary.histogram('bias', b1)
        tf.summary.histogram('activation', h1)
    with tf.name_scope('layer2'):
        tf.summary.histogram('weight', W2)
        tf.summary.histogram('bias', b2)
        tf.summary.histogram('activation', sigmoid_outputs)

    # Merge summary operations
    merged_summary_op = tf.summary.merge_all()


    run_info = utils.init_run_info(NN_OUTPUT_DIM)

    # Experiment info
    run_info['experiment_name'] = experiment_name

    # Problem info
    run_info['operator'] = operator
    run_info['operand_bits'] = operand_bits
    run_info['result_bits'] = target_train.shape[1]

    # Network info
    run_info['network_input_dimension'] = input_train.shape[1]
    run_info['network_output_dimension'] = target_train.shape[1]
    run_info['hidden_activation'] = str_activation
    run_info['hidden_dimensions'] = h_layer_dims

    # Dataset info
    run_info['train_set_size'] = input_train.shape[0]
    run_info['dev_set_size'] = input_dev.shape[0]
    run_info['test_set_size'] = input_test.shape[0]

    # Optimizer info
    run_info['batch_size'] = batch_size
    run_info['optimizer'] = train_op.name
    run_info['learning_rate'] = learning_rate
    run_info['all_correct_stop'] = all_correct_stop

    run_id = datetime.now().strftime('%Y%m%d%H%M%S')
    run_info['run_id'] = run_id

    # Train logging
    logdir = '{}/{}/{}_{}bit_{}_{}_h{}_run-{}/'.format(
        config.dir_logs(), experiment_name, operator, operand_bits, nn_model_type, str_activation, h_layer_dims, run_id)

    train_summary_writer = tf.summary.FileWriter(logdir + '/train', graph=tf.get_default_graph())
    dev_summary_writer = tf.summary.FileWriter(logdir + '/dev')
    if tlu_on:
        tlu_summary_writer = tf.summary.FileWriter(logdir + '/tlu')
    test_summary_writer = tf.summary.FileWriter(logdir + '/test')
    if operator in config.operators_list():
        carry_datasets_summary_writers = create_carry_datasets_summary_writers(logdir, carry_datasets)

    # Model saving
    dir_saved_model = '{}/{}/{}_{}bit_{}_{}_h{}/run-{}/'.format(
        config.dir_saved_models(), experiment_name, operator, operand_bits, nn_model_type, str_activation, h_layer_dims, run_id)
    utils.create_dir(dir_saved_model)

    model_saver = tf.train.Saver()
    init_all_correct_model_saver = tf.train.Saver()

    # Compute nodes
    train_compute_nodes = [loss, op_accuracy, merged_summary_op]
    dev_compute_nodes = [loss, op_accuracy, merged_summary_op, op_wrong, per_digit_accuracy, per_digit_wrong]
    test_compute_nodes = [loss, op_accuracy, merged_summary_op, op_wrong]

    # Session configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    print("Run ID: {}".format(run_id))
    print(logdir)
    print(dir_saved_model)

    with tf.Session(config=tf_config) as sess:
        sess.run(init)

        float_epoch = 0.0
        all_correct_val = False
        big_batch_training_val = False
        init_all_correct_model_saved = False

        for epoch in range(n_epoch):
            input_train, target_train = utils.shuffle_np_arrays(input_train, target_train)

            if big_batch_saturation and all_correct_val:
                big_batch_training_val = True
                batch_size = big_batch_size

            for i_batch in range(n_batch):
                # Get mini-batch
                batch_input, batch_target = utils.get_batch(i_batch, batch_size, input_train, target_train)

                # Initial state evalutation: No training
                if epoch == 0 and i_batch == 0:
                    step = 0
                    float_epoch = 0.0

                    write_train_summary(sess, train_compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step)
                    write_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)
                    if tlu_on:
                        write_tlu_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)


                # Set step, float_epoch
                ## 1 <= (i_batch + 1) <= n_batch
                step = n_batch * epoch + (i_batch + 1)
                float_epoch = epoch + float(i_batch + 1) / n_batch

                # Training operation ##################################################################
                train(sess, batch_input, batch_target, float_epoch, all_correct_val)

                # training set summary writer###########################################################
                if step % train_summary_period == 0:
                    (train_loss, train_accuracy) = write_train_summary(sess, train_compute_nodes, batch_input, batch_target, float_epoch, all_correct_val, step)

                # Development loss evalution
                # After dev_summary_period batches are trained
                if (step % dev_summary_period == 0) or is_last_batch(i_batch):
                    # dev set summary writer#############################################################
                    dev_run_outputs = (dev_loss_val, dev_accuracy_val, dev_op_wrong_val, per_digit_accuracy_val, per_digit_wrong_val) = write_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)

                    # carry datasets summary writer #####################################################
                    if operator in config.operators_list():
                        carry_run_outputs = write_carry_datasets_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)


                    # TLU-dev summary writer#############################################################
                    # tlu_on
                    if tlu_on:
                        dev_tlu_run_outputs = (dev_loss_tlu_val, dev_accuracy_tlu_val, dev_op_wrong_tlu_val) = write_tlu_dev_summary(sess, dev_compute_nodes, float_epoch, all_correct_val, step)
                    else:
                        dev_tlu_run_outputs = None

                    # Write running information################################
                    if operator in config.operators_list():
                        run_info = utils.write_run_info(run_info, float_epoch,
                                            dev_run_outputs, dev_tlu_run_outputs, carry_run_outputs)
                    else:
                        run_info = utils.write_run_info(run_info, float_epoch,
                                            dev_run_outputs, dev_tlu_run_outputs)

                    # Write the logs of measures################################
                    utils.write_measures(run_info, float_epoch,
                                            dev_run_outputs, dev_tlu_run_outputs)


                    if is_last_batch(i_batch):
                        # After one epoch is trained
                        # Save the trained model ################################################
                        model_saver.save(sess, '{}/dev-{}.ckpt'.format(dir_saved_model, run_id))
                        ##print("Model saved.")
                        # decrease_dev_summary_period

                    decrease_dev_summary_period(dev_accuracy_val, dev_op_wrong_val)

                    # If there is no wrong operation, then ...
                    all_correct_val = get_all_correct_val(dev_op_wrong_val)

                    # If the model is  trained with 100% accuracy,
                    if all_correct_val and (not init_all_correct_model_saved):
                        # Save the model.
                        model_name = 'epoch{}-batch{}'.format(float_epoch, i_batch)
                        init_all_correct_model_saver.save(sess, '{}/{}-init-all-correct.ckpt'.format(
                            dir_saved_model, model_name))
                        write_embeddings_summary(sess, h1)
                        init_all_correct_model_saved = True

                    if all_correct_val and all_correct_stop:
                        break # Break the batch for-loop

            # End of one epoch
            if all_correct_val and all_correct_stop:
                break # Break the epoch for-loop

        # End of all epochs

        # Test loss evalution
        # Run computing test loss, accuracy
        # test set summary writer#############################################################
        (test_loss, test_accuracy, test_op_wrong_val) = write_test_summary(sess, test_compute_nodes, float_epoch, all_correct_val, step)

        model_saver.save(sess, '{}/{}.ckpt'.format(dir_saved_model, run_id))
        print("Model saved.")

    # Write running information################################
    if operator in config.operators_list():
        run_info = utils.write_run_info(run_info, float_epoch,
                            dev_run_outputs, dev_tlu_run_outputs, carry_run_outputs)
    else:
        run_info = utils.write_run_info(run_info, float_epoch,
                            dev_run_outputs, dev_tlu_run_outputs)

    train_summary_writer.close()
    dev_summary_writer.close()
    if tlu_on:
        tlu_summary_writer.close()
    test_summary_writer.close()
    if operator in config.operators_list():
        close_carry_datasets_summary_writers(carry_datasets_summary_writers)

    print("The training is over.")


if __name__ == "__main__":
    # execute only if run as a script
    main()
