import datetime
import json
import os
import random
from functools import partial
import numpy as np
import tensorflow as tf
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from RxnPred.configs import Config
from RxnPred.model import RxnPredModel, RxnPredDataset


# set seed
def setSeed(seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def getModelInputs(batch, is_structure=True, is_reaction=True):
    N1 = batch['node1_features']
    E1 = batch['edge1_features']
    N2 = batch['node2_features']
    E2 = batch['edge2_features']
    inputs = [N1, E1, N2, E2]
    if is_structure:
        fp1 = batch['fp1'][:, 0, :]
        fp2 = batch['fp2'][:, 0, :]
        sim = batch['similarity']
        inputs.extend([fp1, fp2, sim])
    if is_reaction:
        rxn = batch['reaction'][:, 0, :]
        inputs.append(rxn)
    inputs = [tf.cast(i, dtype=tf.float32) for i in inputs]
    return inputs


def trainModel(
        model,
        train_dataset,
        valid_dataset=None,
        epochs=100,
        verbose=1,
        optimizer=tf.keras.optimizers.Adam(),
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
):
    """
    Function for model training

    :param model: RxnPred model
    :param train_dataset: training dataset
    :param valid_dataset: validation dataset
    :param epochs: epochs of model training
    :param optimizer: optimizer for model training, default Adam.
    :param loss_fn: loss function used for training, default SparseCategoricalCrossentropy.
    """

    # logger module
    log_dir = "./experiments/logs"
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    train_writer = tf.summary.create_file_writer(log_dir + '/' + current_time + '/train')
    valid_writer = tf.summary.create_file_writer(log_dir + '/' + current_time + '/valid')

    # optimizer and metrics
    # optimizer.build(var_list=model.trainable_variables)  # may influence bayes optimization
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    train_batches = train_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()
    if valid_dataset is not None:
        valid_batches = valid_dataset.reduce(np.int64(0), lambda x, _: x + 1).numpy()

    # training the model
    print('Training Model...')
    print('EPOCHS: ', epochs)
    for epoch in range(epochs):
        # Reset metrics
        train_loss.reset_states()
        valid_loss.reset_states()
        train_accuracy.reset_states()
        valid_accuracy.reset_states()
        # training phase
        if verbose == 1:
            pbar_train = tqdm(total=train_batches, bar_format='{l_bar}{bar:10}{r_bar}', dynamic_ncols=False)
            pbar_train.set_description(f'Epoch {epoch}')
        step = 0
        for batch in train_dataset:
            inputs = getModelInputs(batch)
            labels = batch['label']
            labels = tf.cast(labels, dtype=tf.int32)
            with tf.GradientTape() as tape:
                predictions = model(inputs, training=True)
                loss = loss_fn(labels, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss.update_state(loss)
            train_accuracy.update_state(labels, predictions)
            step = step + 1
            if step == train_batches:
                with train_writer.as_default():
                    tf.summary.scalar('Loss', train_loss.result(), step=epoch)
                    tf.summary.scalar('Accuracy', train_accuracy.result(), step=epoch)
            if verbose == 1:
                pbar_train.set_postfix_str(
                    f'Training Loss: {train_loss.result().numpy():.4f}, '
                    f'Training Accuracy: {train_accuracy.result().numpy():.4f}'
                )
                pbar_train.update()
        # validation phase
        if valid_dataset is not None:
            if verbose == 1:
                pbar_valid = tqdm(total=valid_batches, bar_format='{l_bar}{bar:10}{r_bar}', dynamic_ncols=False)
                pbar_valid.set_description(f'Epoch {epoch}')
            step = 0
            for batch in valid_dataset:
                inputs = getModelInputs(batch)
                labels = batch['label']
                labels = tf.cast(labels, dtype=tf.int32)
                predictions = model(inputs, training=False)
                loss = loss_fn(labels, predictions)
                valid_loss.update_state(loss)
                valid_accuracy.update_state(labels, predictions)
                step = step + 1
                if step == valid_batches:
                    with valid_writer.as_default():
                        tf.summary.scalar('Loss', valid_loss.result(), step=epoch)
                        tf.summary.scalar('Accuracy', valid_accuracy.result(), step=epoch)
                if verbose == 1:
                    pbar_valid.set_postfix_str(
                        f'Validation Loss: {valid_loss.result().numpy():.4f}, '
                        f'Validation Accuracy: {valid_accuracy.result().numpy():.4f}'
                    )
                    pbar_valid.update()
    print('Training Model OK!')


def trainFunction(
        filename_train,
        filename_valid,
        config=Config(),
        is_save=True,
        verbose=1,
        **parameters
):
    """
    Train a model. [For bayes optimization]

    :return: Validation Accuracy.
    """

    # load parameters for model
    model_params = [
        'batch_size',
        'num_epochs',
        'learning_rate',
        'num_gconv_layers',
        'num_gconv_units',
        'num_dense_layers',
        'num_dense_units',
        'weight_decay',
        'dense_dropout'
    ]
    for para in model_params:
        if para in parameters:
            if para == 'batch_size':
                config[para] = 16 * int(parameters[para])
            elif para in {'dense_dropout', 'learning_rate', 'weight_decay'}:
                config[para] = parameters[para]
            else:
                config[para] = int(parameters[para])
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate
    num_gconv_layers = config.num_gconv_layers
    num_gconv_units = config.num_gconv_units
    num_dense_layers = config.num_dense_layers
    num_dense_units = config.num_dense_units
    weight_decay = config.weight_decay
    dense_dropout = config.dense_dropout
    params = {
        "gconv_units": [num_gconv_units] * num_gconv_layers,
        "gconv_regularizer": tf.keras.regularizers.L2(weight_decay),
        'dense_units': [num_dense_units] * num_dense_layers,
        'dense_dropout': dense_dropout,
    }
    model = RxnPredModel(**params)

    # load datasets and train model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=1)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    train_dataset = RxnPredDataset(filenames=filename_train, batch_size=batch_size, training=True)
    train_dataset = train_dataset.get_iterator()
    valid_dataset = RxnPredDataset(filenames=filename_valid, batch_size=batch_size, training=False)
    valid_dataset = valid_dataset.get_iterator()
    trainModel(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        epochs=num_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        verbose=verbose
    )

    # calculate error and return
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
    valid_accuracy.reset_states()
    for batch in valid_dataset:
        inputs = getModelInputs(batch)
        labels = batch['label']
        labels = tf.cast(labels, dtype=tf.int32)
        predictions = model(inputs, training=False)
        valid_accuracy.update_state(labels, predictions)
    # save model weights
    if is_save:
        Performance = {
            'valid_accuracy': valid_accuracy.result().numpy(),
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "initial_learning_rate": learning_rate,
            'num_gconv_layers': num_gconv_layers,
            "num_gconv_units": num_gconv_units,
            "num_dense_layers": num_dense_layers,
            "num_dense_units": num_dense_units,
            "weight_decay": weight_decay,
            'dense_dropout': dense_dropout
        }
        json_object = json.dumps(Performance, cls=NpEncoder)
        save_path = config.save_path
        now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_weights(os.path.join(save_path, "model_{}.ckpt".format(now_time)))
        with open(save_path + "Performance_" + str(now_time) + ".json", "w") as outfile:
            outfile.write(json_object)
    return valid_accuracy.result().numpy()


def bayesHyperParamSearch(filename_train, filename_valid, number_search=100):
    """
    Using Bayes hyperparameter search to search the best parameters
    :param number_search: rounds of Bayes hyperparameter search
    """
    train_partial = partial(
        trainFunction,
        filename_train=filename_train,
        filename_valid=filename_valid
    )
    # Bounded region of parameter space
    pbounds = {
        'batch_size': (2, 8),  # n*16
        'num_epochs': (5, 26),
        'learning_rate': (1e-5, 1e-3),
        'num_gconv_layers': (2, 4),
        'num_gconv_units': (128, 257),
        'num_dense_layers': (2, 4),
        'num_dense_units': (128, 513),
        'weight_decay': (1e-06, 1e-03),
        # 'dense_dropout': (0.0, 0.1),
    }
    optimizer = BayesianOptimization(
        f=train_partial,
        pbounds=pbounds,
        verbose=2,
        random_state=42,
    )
    optimizer.maximize(init_points=10, n_iter=number_search)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


if __name__ == '__main__':
    setSeed(seed=42)
    filename_train = './RxnPred/rp_data_train.tfrecord'  # for test
    filename_valid = './RxnPred/rp_data_valid.tfrecord'
    bayesHyperParamSearch(
        filename_train=filename_train,
        filename_valid=filename_valid,
        number_search=50
    )
    # config = Config()
    # config = config.load(filepath='./RxnPred/default_configs.json')
    # trainFunction(
    #     filename_train=filename_train,
    #     filename_valid=filename_valid,
    #     config=config,
    #     is_save=True,
    #     verbose=1
    # )
