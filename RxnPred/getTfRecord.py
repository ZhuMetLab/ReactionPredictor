import numpy as np
import pandas as pd
import tensorflow as tf
from rdkit import Chem
from rdkit.Chem import AllChem

from RxnPred.ChemClean import MolCleaner
from RxnPred import getGraphFeature
from RxnPred.getReactionClass import reaction_class
from RxnPred.processModification import formula2dict, modification2dict


def getGraphData(smiles):

    def _getSparseData(arr):
        indices = tf.cast(tf.where(arr != 0), tf.int64)
        values = tf.cast(tf.gather_nd(arr, indices), tf.float32)
        shape = tf.cast(tf.shape(arr), tf.int64)
        return indices.numpy(), values.numpy(), shape.numpy()

    mol = Chem.MolFromSmiles(smiles)
    node_features = getGraphFeature.getNodeFeatures(mol)
    edge_features = getGraphFeature.getEdgeFeatures(mol)
    node_indices, node_values, node_shape = _getSparseData(node_features)
    edge_indices, edge_values, edge_shape = _getSparseData(edge_features)
    return {
        'node_indices': node_indices,
        'node_values': node_values,
        'node_shape': node_shape,
        'edge_indices': edge_indices,
        'edge_values': edge_values,
        'edge_shape': edge_shape
    }


def GetTfRecord(
    dataframe: pd.DataFrame,
    save_name: str = 'demo',
    is_label: bool = False,
    is_preset: bool = True,
    col_label: str = 'score',
    rxn_class: str = './RxnPred/ref_rxn_class.csv'
):

    if is_preset:
        print('Generate TFRecord for training/validation/testing datasets.')
        df_train = dataframe[dataframe['set'] == 'train'].reset_index(drop=True)
        df_valid = dataframe[dataframe['set'] == 'valid'].reset_index(drop=True)
        df_test = dataframe[dataframe['set'] == 'test'].reset_index(drop=True)
        print('Training dataset...')
        GetTfRecord(dataframe=df_train, save_name=save_name + '_train', is_label=True, is_preset=False)
        print('Training dataset OK!')
        print('Validation dataset...')
        GetTfRecord(dataframe=df_valid, save_name=save_name + '_valid', is_label=True, is_preset=False)
        print('Validation dataset OK!')
        print('Testing dataset...')
        GetTfRecord(dataframe=df_test, save_name=save_name + '_test', is_label=False, is_preset=False)
        print('Testing dataset OK!')
        return

    # get labels/targets
    if is_label:
        label_all = dataframe[col_label]
    else:
        label_all = dataframe.index

    # get index
    index_all = dataframe.index

    # sort SMILES and get index for SMILES -----------------------------------------------------------------------------
    sorted_smiles = dataframe.apply(lambda row: sorted([row['SMILES1'], row['SMILES2']]), axis=1)
    sorted_dataframe = pd.DataFrame(sorted_smiles.to_list(), columns=['SMILES1', 'SMILES2'])
    dataframe['SMILES1'] = sorted_dataframe['SMILES1']
    dataframe['SMILES2'] = sorted_dataframe['SMILES2']
    smiles1_all = dataframe['SMILES1']
    smiles2_all = dataframe['SMILES2']
    smiles_set = set(smiles1_all) | set(smiles2_all)
    smiles_list = sorted(list(smiles_set))
    idx1 = [smiles_list.index(smiles) for smiles in smiles1_all]
    idx2 = [smiles_list.index(smiles) for smiles in smiles2_all]

    # get graph features -----------------------------------------------------------------------------------------------
    print('Get graph features...')
    graph_data = [getGraphData(smiles) for smiles in smiles_list]
    graph_data1 = [graph_data[i] for i in idx1]
    graph_data2 = [graph_data[i] for i in idx2]
    print('Get graph features OK!')

    # get FP and Tanimoto similarity -----------------------------------------------------------------------------------
    print('Get fingerprint features...')
    fp_data = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(smiles),
            radius=2, nBits=1024, useFeatures=False, useChirality=False
        ) for smiles in smiles_list
    ]
    fp1_data = np.array([np.array(fp_data[i].ToList()) for i in idx1])
    fp2_data = np.array([np.array(fp_data[i].ToList()) for i in idx2])
    similarity_data = np.array([
        AllChem.DataStructs.TanimotoSimilarity(fp_data[i1], fp_data[i2])
        for i1, i2 in zip(idx1, idx2)
    ])
    # # version 1 (slow)
    # fp1_all = [
    #     AllChem.GetMorganFingerprintAsBitVect(
    #         Chem.MolFromSmiles(smiles1),
    #         radius=2, nBits=1024, useFeatures=False, useChirality=False
    #     ) for smiles1 in smiles1_all
    # ]
    # fp1_data = np.array([fp1.ToList() for fp1 in fp1_all])
    # fp2_all = [
    #     AllChem.GetMorganFingerprintAsBitVect(
    #         Chem.MolFromSmiles(smiles2),
    #         radius=2, nBits=1024, useFeatures=False, useChirality=False
    #     ) for smiles2 in smiles2_all
    # ]
    # fp2_data = np.array([fp2.ToList() for fp2 in fp2_all])
    # similarity_data = np.array([
    #     AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    #     for fp1, fp2 in zip(fp1_all, fp2_all)
    # ])
    print('Get fingerprint features OK!')

    # get reaction features --------------------------------------------------------------------------------------------
    print('Get reaction features...')
    # set reaction_class_data 0 (n, 2)
    reaction_class_data = np.zeros((len(dataframe), 2)).astype(int)
    # load reference reaction data
    df_rxn_class = pd.read_csv(rxn_class)
    list_atom_diff = df_rxn_class['atom_diff'].to_list()
    rxnclass_value = np.array([reaction_class.get(i) for i in list(df_rxn_class['reaction_class'])])
    # reaction class "others": 0
    rxnclass_value[rxnclass_value == 7] = 0
    freq_value = np.array(list(df_rxn_class['count']))
    # create reference atom matrix
    atom_list = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'Cl', 'Br', 'I', 'Se', 'B']
    atom_diff_matrix = np.zeros((len(atom_list), len(list_atom_diff))).astype(int)
    dict_atom_diff = [modification2dict(x) for x in list_atom_diff]
    for i, atom_diff in enumerate(dict_atom_diff):
        for atom, count in atom_diff.items():
            atom_diff_matrix[atom_list.index(atom), i] = count
    # create smiles1 and smiles2 atom matrix
    molCleaner = MolCleaner()
    smiles1_atom_matrix = np.zeros((len(atom_list), len(smiles1_all))).astype(int)
    smiles2_atom_matrix = np.zeros((len(atom_list), len(smiles2_all))).astype(int)
    smiles_atom_dict = [formula2dict(molCleaner.smiles2formula(smiles, is_clean=False)) for smiles in smiles_list]
    for i, i1 in enumerate(idx1):
        for atom, count in smiles_atom_dict[i1].items():
            smiles1_atom_matrix[atom_list.index(atom), i] = count
    for i, i2 in enumerate(idx2):
        for atom, count in smiles_atom_dict[i2].items():
            smiles2_atom_matrix[atom_list.index(atom), i] = count
    # get smiles1 and smiles2 atom difference matrix (+/-), map to reaction_class_data [type, freq.] (n, 2)
    df_atom_diff_matrix = smiles1_atom_matrix - smiles2_atom_matrix
    check_equal1 = np.all(atom_diff_matrix[:, np.newaxis, :] == df_atom_diff_matrix[:, :, np.newaxis], axis=0)
    check_equal2 = np.all(atom_diff_matrix[:, np.newaxis, :] == -df_atom_diff_matrix[:, :, np.newaxis], axis=0)
    rows1, cols1 = np.where(check_equal1)
    rows2, cols2 = np.where(check_equal2)
    reaction_class_data[rows1, 0] = rxnclass_value[cols1]
    reaction_class_data[rows1, 1] = freq_value[cols1]
    reaction_class_data[rows2, 0] = rxnclass_value[cols2]
    reaction_class_data[rows2, 1] = freq_value[cols2]
    # # version 1 (slow)
    # dict_atom2rxnclass, dict_atom2count = getReactionClass.getDictAtomDiff2RxnClass()
    # atom_diff = [
    #     getReactionClass.getAtomDiffBySmiles(smiles1=smiles1, smiles2=smiles2)
    #     for smiles1, smiles2 in zip(smiles1_all, smiles2_all)
    # ]
    # reaction_class_data = np.array([
    #     [
    #         reaction_class.get(dict_atom2rxnclass.get(atom_diff)),
    #         dict_atom2count.get(atom_diff, 0)
    #     ] for atom_diff in atom_diff
    # ])
    print('Get reaction features OK!')

    print('Write to TFRecord...')

    def getDataDict(inputs):
        index, label, smiles1, smiles2, fp1, fp2, similarity, reaction, graph1, graph2 = inputs
        return {
            'index': index,
            'label': label,
            'smiles1': smiles1,
            'smiles2': smiles2,
            'fp1': fp1,
            'fp2': fp2,
            'similarity': similarity,
            'reaction': reaction,
            'node1_indices': graph1['node_indices'],
            'node1_values': graph1['node_values'],
            'node1_shape': graph1['node_shape'],
            'edge1_indices': graph1['edge_indices'],
            'edge1_values': graph1['edge_values'],
            'edge1_shape': graph1['edge_shape'],
            'node2_indices': graph2['node_indices'],
            'node2_values': graph2['node_values'],
            'node2_shape': graph2['node_shape'],
            'edge2_indices': graph2['edge_indices'],
            'edge2_values': graph2['edge_values'],
            'edge2_shape': graph2['edge_shape']
        }

    data_dict_list = [
        getDataDict(inputs=i) for i in zip(
            index_all, label_all, smiles1_all, smiles2_all,
            fp1_data, fp2_data, similarity_data,
            reaction_class_data,
            graph_data1, graph_data2
        )
    ]

    data_dict = {k: [d[k] for d in data_dict_list] for k in data_dict_list[0]}

    def _serializeFeature(value):
        value = tf.io.serialize_tensor(value).numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    with tf.io.TFRecordWriter(save_name + '.tfrecord') as writer:
        for i in range(len(data_dict['index'])):
            features_dict = {}
            for key, value in data_dict.items():
                if key == 'index':
                    value_to_store = _serializeFeature(np.array(value[i], dtype='int64'))
                elif key == 'label':
                    value_to_store = _serializeFeature(np.array(value[i], dtype='int64'))
                elif isinstance(value[i], np.ndarray):
                    value_to_store = _serializeFeature(value[i])
                else:
                    value_to_store = _serializeFeature(np.array(value[i]))
                features_dict[key] = value_to_store
            example = tf.train.Example(features=tf.train.Features(feature=features_dict))
            writer.write(example.SerializeToString())
    print('Write to TFRecord OK!')


if __name__ == '__main__':

    dataframe = pd.read_csv('./rp_data_for_training.csv')
    GetTfRecord(dataframe=dataframe, save_name='rp_data', is_preset=True)
