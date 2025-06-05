import numpy as np
import tensorflow as tf


class RxnPredDataset:
    def __init__(
        self,
        filenames: str = 'data_train.tfrecord',
        batch_size: int = 128,
        training: bool = False,
        random_seed: int = 42,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    ):
        self.filenames = filenames
        self.batch_size = batch_size
        self.training = training
        self.random_seed = random_seed
        self.num_parallel_calls = num_parallel_calls

    def __len__(self):
        dataset = tf.data.TFRecordDataset(
            filenames=self.filenames,
            num_parallel_reads=self.num_parallel_calls).repeat(1)
        return dataset.reduce(np.int64(0), lambda x, _: x + 1)

    @property
    def padded_shapes(self):
        return {
            'index': (None,),
            'label': (None,),
            'smiles1': (None,),
            'smiles2': (None,),
            'fp1': (None, None),
            'fp2': (None, None),
            'similarity': (None,),
            'reaction': (None, None),
            'node1_features': (None, None,),
            'edge1_features': (None, None, None,),
            'node2_features': (None, None,),
            'edge2_features': (None, None, None,),
        }

    @staticmethod
    def preprocess_function(features: dict):
        edge1_features = tf.scatter_nd(
            indices=features['edge1_indices'],
            updates=features['edge1_values'],
            shape=features['edge1_shape']
        )
        node1_features = tf.scatter_nd(
            indices=features['node1_indices'],
            updates=features['node1_values'],
            shape=features['node1_shape']
        )
        edge2_features = tf.scatter_nd(
            indices=features['edge2_indices'],
            updates=features['edge2_values'],
            shape=features['edge2_shape']
        )
        node2_features = tf.scatter_nd(
            indices=features['node2_indices'],
            updates=features['node2_values'],
            shape=features['node2_shape']
        )

        return {
            'index': [features['index']],
            'label': [features['label']],
            'smiles1': [features['smiles1']],
            'smiles2': [features['smiles2']],
            'fp1': [features['fp1']],
            'fp2': [features['fp2']],
            'similarity': [features['similarity']],
            'reaction': [features['reaction']],
            'node1_features': node1_features,
            'edge1_features': edge1_features,
            'node2_features': node2_features,
            'edge2_features': edge2_features,
        }

    @staticmethod
    def _parse_function(examples):
        features = [
            ('index', tf.int64),
            ('label', tf.int64),
            ('smiles1', tf.string),
            ('smiles2', tf.string),
            ('fp1', tf.int32),
            ('fp2', tf.int32),
            ('similarity', tf.double),
            ('reaction', tf.int32),
            ('node1_indices', tf.int64),
            ('node1_values', tf.float32),
            ('node1_shape', tf.int64),
            ('edge1_indices', tf.int64),
            ('edge1_values', tf.float32),
            ('edge1_shape', tf.int64),
            ('node2_indices', tf.int64),
            ('node2_values', tf.float32),
            ('node2_shape', tf.int64),
            ('edge2_indices', tf.int64),
            ('edge2_values', tf.float32),
            ('edge2_shape', tf.int64),
        ]
        feature_descriptions = {
            f[0]: tf.io.FixedLenFeature([], tf.string)
            for f in features
        }
        example = tf.io.parse_single_example(examples, feature_descriptions)
        features = {
            f[0]: tf.io.parse_tensor(example[f[0]], out_type=f[1])
            for f in features
        }
        return features

    def _get_parsed_tfrecord_dataset(self) -> tf.data.Dataset:
        dataset = tf.data.TFRecordDataset(
                filenames=self.filenames,
                num_parallel_reads=self.num_parallel_calls
        ).repeat(1)
        dataset = dataset.map(self._parse_function, self.num_parallel_calls)
        if self.training:
            dataset = dataset.shuffle(len(self), seed=self.random_seed)
        return dataset

    def get_iterator(self) -> tf.data.Dataset:
        dataset = self._get_parsed_tfrecord_dataset()
        dataset = dataset.map(self.preprocess_function, self.num_parallel_calls)
        dataset = dataset.padded_batch(self.batch_size, padded_shapes=self.padded_shapes)
        return dataset


if __name__ == '__main__':

    dataset_test = RxnPredDataset(filenames='rp_data_test.tfrecord', batch_size=32, training=False)
    dataset_test = dataset_test.get_iterator()
