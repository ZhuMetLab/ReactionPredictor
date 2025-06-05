import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from RxnPred.processModification import getAtomDiffByFormula
from RxnPred.ChemClean import MolCleaner


class ColumnNotFoundError(Exception):
    pass


# get atom difference
def getAtomDiffBySmiles(smiles1, smiles2):
    Cleaner = MolCleaner()
    smiles1 = Cleaner.cleanSmiles(smiles=smiles1)
    smiles2 = Cleaner.cleanSmiles(smiles=smiles2)
    formula1 = CalcMolFormula(Chem.MolFromSmiles(smiles1))
    formula2 = CalcMolFormula(Chem.MolFromSmiles(smiles2))
    return getAtomDiffByFormula(formula1, formula2)


# Reaction class information
def getDictAtomDiff2RxnClass(
    rxn_class: str = './RxnPred/ref_rxn_class.csv',
    is_rm_others: bool = True
):
    df_rxn_class = pd.read_csv(rxn_class)
    # check column name
    check_list = ['atom_diff', 'reaction_class', 'count']
    missing = [col for col in check_list if col not in df_rxn_class.columns]
    if missing:
        raise ColumnNotFoundError(
            f"The following column does not exist in the DataFrame: {', '.join(missing)}"
        )
    dict_atom2count = dict(zip(df_rxn_class['atom_diff'], df_rxn_class['count']))
    if is_rm_others:
        df_rxn_class = df_rxn_class[~(df_rxn_class['reaction_class'] == 'Others')]
    dict_atom2rxnclass = dict(zip(df_rxn_class['atom_diff'], df_rxn_class['reaction_class']))
    return dict_atom2rxnclass, dict_atom2count


reaction_class = {
    "Redox reaction": 1,
    "Transfer reaction": 2,
    "Lytic reaction": 3,
    "Hydrolysis": 4,
    "Isomerazation": 5,
    "Ligation reaction": 6,
    "Others": 7,
    None: 0
}

if __name__ == '__main__':

    dict_atom2rxnclass, dict_atom2count = getDictAtomDiff2RxnClass()
    smiles1 = 'NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O'
    smiles2 = 'NC1=NC=NC2=C1N=CN2[C@@H]1O[C@H](COP(=O)(O)OS(=O)(=O)O)[C@@H](O)[C@H]1O'
    atom_diff = getAtomDiffBySmiles(smiles1=smiles1, smiles2=smiles2)
    reaction_class_info = np.array([
        reaction_class.get(dict_atom2rxnclass.get(atom_diff)),
        dict_atom2count.get(atom_diff, 0)
    ])
    print('reaction class feature shape:', reaction_class_info.shape)
    print('reaction class feature:', reaction_class_info)

    smiles1 = 'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS'
    smiles2 = 'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCCC(=O)O'
    atom_diff = getAtomDiffBySmiles(smiles1=smiles1, smiles2=smiles2)
    reaction_class_info = np.array([
        reaction_class.get(dict_atom2rxnclass.get(atom_diff)),
        dict_atom2count.get(atom_diff, 0)
    ])
    print('reaction class feature shape:', reaction_class_info.shape)
    print('reaction class feature:', reaction_class_info)
