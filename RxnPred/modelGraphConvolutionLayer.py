import tensorflow as tf


class GraphConvolutionLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        num_bases=-1,
        units=128,
        activation='relu',
        dropout=0.0,
        batch_norm=False,
        initializer='glorot_uniform',
        regularizer=None,
        **kwargs
    ):

        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.num_bases = num_bases
        self.units = units
        self.dropout = (
            tf.keras.layers.Dropout(dropout) if dropout
            else tf.keras.layers.Lambda(lambda x: x)
        )
        self.batch_norm = (
            tf.keras.layers.BatchNormalization() if batch_norm
            else tf.keras.layers.Lambda(lambda x: x)
        )
        self.activation = tf.keras.activations.get(activation)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.initializer = tf.keras.initializers.get(initializer)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask[0]

    def build(self, input_shape):

        assert isinstance(input_shape, list)
        self.num_bases = input_shape[1][-1]
        assert self.num_bases > 0

        self.Wi = self.add_weight(
            name='Wi',
            shape=(input_shape[0][-1], self.units, self.num_bases),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            dtype=tf.float32
        )
        self.W0 = self.add_weight(
            name='W0',
            shape=(input_shape[0][-1], self.units),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            dtype=tf.float32
        )

    def call(self, inputs, training=False, mask=None):

        (N, E) = inputs
        N0 = tf.matmul(N, self.W0)

        for i in range(E.shape[-1]):
            Ei = E[..., i]
            Ni = tf.matmul(N, self.Wi[..., i])
            Ni = tf.matmul(Ei, Ni)
            N0 = N0 + Ni

        N = N0

        N = self.batch_norm(N, training=training)
        N = self.activation(N)
        N = self.dropout(N, training=training)
        if mask:
            N_mask = mask[0][:, :, None]
            N *= tf.cast(N_mask, N.dtype)
        return N
