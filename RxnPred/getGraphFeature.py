import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors, rdPartialCharges
from RxnPred.ChemClean import MolCleaner

RDLogger.DisableLog('rdApp.*')

# node_dim = 20
# edge_dim = 5
# adj_dim = 1


def getNodeFeatures(mol):

    node_features = np.array([
        getAtomFeatures(atom) for atom in mol.GetAtoms()
    ], dtype='float32')
    node_features[np.isnan(node_features)] = 0
    node_features[np.isinf(node_features)] = 0

    return node_features


def getEdgeFeatures(mol):

    adj_dim = 1
    edge_dim = 5
    edge_features = np.zeros(
        shape=(mol.GetNumAtoms(), mol.GetNumAtoms(), adj_dim + edge_dim),
        dtype='float32'
    )
    for atom in mol.GetAtoms():
        a = atom.GetIdx()
        for neighbor in atom.GetNeighbors():
            i = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(a, i)
            bond_features = getBondFeatures(bond)
            edge_features[[a, i], [i, a], :] = bond_features[np.newaxis, :]

    # remove constant features (edge features)
    edge_min = np.min(edge_features, axis=(0, 1))
    edge_max = np.max(edge_features, axis=(0, 1))
    idx_normalized = np.where(edge_max - edge_min != 0.0)[0]

    # normalize edge features matrix
    np.seterr(divide='ignore', invalid='ignore')
    for i in idx_normalized:
        adjacency_matrix = edge_features[:, :, i]
        R = np.sum(adjacency_matrix, axis=1)
        D_sqrt = np.nan_to_num(np.diag(1/np.sqrt(R)))
        edge_features[:, :, i] = np.matmul(np.matmul(D_sqrt, adjacency_matrix), D_sqrt)

    return edge_features


allowable_features = {
    'atom_type': [
        'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
        'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    ],
    'chirality': [
        Chem.rdchem.ChiralType.CHI_OTHER,
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ],
    'hybridization': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,  # 20230421 update
    ],
    'is_in_ring_size_n': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0],
    'bond_type': [
        np.NaN,
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        Chem.rdchem.BondType.DATIVE,
    ],
    'bond_stereo': [
        np.NaN,
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOANY,
    ]
}


def getAtomFeatures(atom: Chem.Atom):
    mol = atom.GetOwningMol()
    # atom_type
    atom_type = allowable_features['atom_type'].index(atom.GetSymbol())
    # atom_mass
    atom_mass = atom.GetMass()
    # chiral_center
    chiral_center = atom.HasProp("_ChiralityPossible")
    # chirality
    chirality = allowable_features['chirality'].index(atom.GetChiralTag())
    # degree
    degree = min(atom.GetDegree(), 5)
    # formal_charge
    formal_charge = atom.GetFormalCharge()
    # gasteiger_charge
    rdPartialCharges.ComputeGasteigerCharges(mol)
    gasteiger_charge = atom.GetDoubleProp('_GasteigerCharge')
    # hybridization
    hybridization = allowable_features['hybridization'].index(atom.GetHybridization())
    # is_aromatic
    is_aromatic = atom.GetIsAromatic()
    # is_hydrogen_donor
    is_hydrogen_donor = atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]
    # is_hydrogen_acceptor
    is_hydrogen_acceptor = atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]
    # is_hetero
    is_hetero = atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]
    # is_in_ring_size_n
    for ring_size in allowable_features['is_in_ring_size_n']:
        if atom.IsInRingSize(ring_size):
            break
    is_in_ring_size_n = ring_size
    # num_hydrogens
    num_hydrogens = min(atom.GetTotalNumHs(), 4)
    # num_radical_electrons
    num_radical_electrons = min(atom.GetNumRadicalElectrons(), 2)
    # num_valence_electrons
    num_valence_electrons = min(atom.GetTotalValence(), 6)
    # crippen_log_p_contribution
    crippen_log_p_contribution = Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
    # crippen_molar_refractivity_contribution
    crippen_molar_refractivity_contribution = Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
    # tpsa_contribution
    tpsa_contribution = rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
    # lasa_contribution
    lasa_contribution = rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
    # summary
    return np.array([
        atom_type,
        atom_mass,
        chiral_center,
        chirality,
        degree,
        formal_charge,
        gasteiger_charge,
        hybridization,
        is_aromatic,
        is_hydrogen_donor,
        is_hydrogen_acceptor,
        is_hetero,
        is_in_ring_size_n,
        num_hydrogens,
        num_radical_electrons,
        num_valence_electrons,
        crippen_log_p_contribution,
        crippen_molar_refractivity_contribution,
        tpsa_contribution,
        lasa_contribution
    ])


def getBondFeatures(bond: Chem.Bond):
    mol = bond.GetOwningMol()
    # is_adjacency
    is_adjacency = 1
    # bond_type
    bond_type = allowable_features['bond_type'].index(bond.GetBondType())
    # bond_stereo
    bond_stereo = allowable_features['bond_stereo'].index(bond.GetStereo())
    # is_conjugated
    is_conjugated = bond.GetIsConjugated()
    # is_in_ring
    is_in_ring = bond.IsInRing()
    # is_rotatable
    atom_indices = tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    is_rotatable = atom_indices in Lipinski._RotatableBonds(mol)
    # summary
    return np.array([
        is_adjacency,
        bond_type,
        bond_stereo,
        is_conjugated,
        is_in_ring,
        is_rotatable,
    ])


if __name__ == '__main__':

    Cleaner = MolCleaner()
    smiles = 'NC1=C2N=CN([C@@H]3O[C@H](COP(=O)(O)O[Se](=O)(=O)O)[C@@H](O)[C@H]3O)C2=NC=N1'
    smiles2 = Cleaner.cleanSmiles(smiles=smiles)
    mol = Chem.MolFromSmiles(smiles2)
    node_features = getNodeFeatures(mol)
    edge_features = getEdgeFeatures(mol)

    print('Input SMILES:', smiles)
    print('Standardized SMILES:', smiles2)
    print('Node feature matrix shape:', node_features.shape)
    print('Edge feature matrix shape:', edge_features.shape)
