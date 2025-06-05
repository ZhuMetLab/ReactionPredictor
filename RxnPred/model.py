import tensorflow as tf
from tqdm import tqdm

from RxnPred.dataset import RxnPredDataset
from RxnPred.modelGraphConvolutionLayer import GraphConvolutionLayer


class RxnPredModel(tf.keras.Model):
    def __init__(
            self,
            gconv_units=None,
            gconv_activation='relu',
            gconv_dropout=0.0,
            gconv_batch_norm=False,
            gconv_initializer='glorot_uniform',
            gconv_regularizer=None,
            dense_units=None,
            dense_activation='relu',
            dense_dropout=0.0,
            dense_initializer='glorot_uniform',
            dense_regularizer=None,
            **kwargs
    ):
        super(RxnPredModel, self).__init__(**kwargs)
        if gconv_units is None:
            gconv_units = [256, 256, 256]
        if dense_units is None:
            dense_units = [2048, 1024, 512, 256, 128, 64]
        self.masking = tf.keras.layers.Masking(mask_value=0)
        self.gconv_layers = [
            GraphConvolutionLayer(
                units=gconv_unit,
                activation=gconv_activation,
                dropout=gconv_dropout,
                batch_norm=gconv_batch_norm,
                initializer=gconv_initializer,
                regularizer=gconv_regularizer,
            )
            for gconv_unit in gconv_units
        ]
        self.pooling = tf.keras.layers.GlobalAveragePooling1D()
        self.dense_layers = [
            tf.keras.layers.Dense(
                units=dense_unit,
                activation=dense_activation,
                kernel_initializer=dense_initializer,
                kernel_regularizer=dense_regularizer
            )
            for dense_unit in dense_units
        ]
        self.dense_dropout = tf.keras.layers.Dropout(dense_dropout)
        self.dense_output = tf.keras.layers.Dense(2)

    def call(self, inputs, training=None, masking=None):

        (N1, E1, N2, E2, fp1, fp2, sim, rxn) = inputs

        N1 = self.masking(N1)
        for i in range(len(self.gconv_layers)):
            N1 = self.gconv_layers[i]([N1, E1])
        N1 = self.pooling(N1)

        N2 = self.masking(N2)
        for i in range(len(self.gconv_layers)):
            N2 = self.gconv_layers[i]([N2, E2])
        N2 = self.pooling(N2)

        X = tf.concat([N1, N2, fp1, fp2, sim, rxn], axis=1)

        for i in range(len(self.dense_layers)):
            X = self.dense_layers[i](X)
            X = self.dense_dropout(X)

        X = self.dense_output(X)

        return X


if __name__ == '__main__':

    dataset_train = RxnPredDataset(filenames='./rp_data_train.tfrecord', batch_size=256, training=True)
    dataset_train = dataset_train.get_iterator()
    dataset_valid = RxnPredDataset(filenames='./rp_data_valid.tfrecord', batch_size=256, training=False)
    dataset_valid = dataset_valid.get_iterator()

    model = RxnPredModel()
    # example
    inputs = [
        tf.keras.Input(shape=(56, 20)),
        tf.keras.Input(shape=(56, 56, 6)),
        tf.keras.Input(shape=(7, 20)),
        tf.keras.Input(shape=(7, 7, 6)),
        tf.keras.Input(shape=1024),
        tf.keras.Input(shape=1024),
        tf.keras.Input(shape=1),
        tf.keras.Input(shape=2),
    ]
    outputs = model(inputs)

    # test training
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.1)
    optimizer.build(var_list=model.trainable_variables)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(inputs, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(inputs, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(inputs, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    model.save_weights("model_test_save_before.ckpt")
    pbar = tqdm(range(20))
    for epoch in pbar:
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        i_batch = 0
        for batch in dataset_train:
            i_batch = i_batch + 1
            # if i_batch < 60:
            #     continue
            labels = batch['label']
            labels = tf.cast(labels, dtype=tf.int32)
            N1 = batch['node1_features']
            E1 = batch['edge1_features']
            N2 = batch['node2_features']
            E2 = batch['edge2_features']
            fp1 = batch['fp1'][:, 0, :]
            fp2 = batch['fp2'][:, 0, :]
            sim = batch['similarity']
            rxn = batch['reaction'][:, 0, :]
            inputs = [N1, E1, N2, E2, fp1, fp2, sim, rxn]
            inputs = [tf.cast(i, dtype=tf.float32) for i in inputs]
            train_step(inputs, labels)
            print(
                f'Epoch {epoch + 1}, '
                f'i_batch: {i_batch}, '
                f'Loss: {train_loss.result():.2f}, '
                f'Accuracy: {train_accuracy.result() * 100:.2f}%'
            )
            if train_loss.result() > 1:
                print(f'{i_batch}' + " [ train loss > 1 ]")
        for batch in dataset_valid:
            labels = batch['label']
            labels = tf.cast(labels, dtype=tf.int32)
            N1 = batch['node1_features']
            E1 = batch['edge1_features']
            N2 = batch['node2_features']
            E2 = batch['edge2_features']
            fp1 = batch['fp1'][:, 0, :]
            fp2 = batch['fp2'][:, 0, :]
            sim = batch['similarity']
            rxn = batch['reaction'][:, 0, :]
            inputs = [N1, E1, N2, E2, fp1, fp2, sim, rxn]
            inputs = [tf.cast(i, dtype=tf.float32) for i in inputs]
            test_step(inputs, labels)
            print(
                f'Epoch {epoch + 1}, '
                f'Validation Loss: {test_loss.result():.2f}, '
                f'Validation Accuracy: {test_accuracy.result() * 100:.2f}%'
            )

    model.save_weights("model_test_after.ckpt")
