from rdkit import Chem
from rdkit.Chem import AllChem
from RxnPred.ChemClean import MolCleaner


def getStructureSimilarityBySmiles(smiles1, smiles2, method: str = 'Tanimoto'):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is not None and mol2 is not None:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=1024, useFeatures=False, useChirality=False)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=1024, useFeatures=False, useChirality=False)
        if method == 'Tanimoto':
            similarity = AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == 'Dice':
            similarity = AllChem.DataStructs.DiceSimilarity(fp1, fp2)
        elif method == 'Cosine':
            similarity = AllChem.DataStructs.CosineSimilarity(fp1, fp2)
        elif method == 'Russel':
            similarity = AllChem.DataStructs.RusselSimilarity(fp1, fp2)
    else:
        similarity = None
    return similarity


if __name__ == '__main__':

    Cleaner = MolCleaner()
    smiles1 = 'C[Se]CCC(C(=O)O)(N)[H]'
    smiles1 = Cleaner.cleanSmiles(smiles=smiles1)
    smiles2 = 'CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](N2C=NC3=C2N=CN=C3N)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS'
    smiles2 = Cleaner.cleanSmiles(smiles=smiles2)
    print('SMILES1:', smiles1)
    print('SMILES2:', smiles2)
    print(getStructureSimilarityBySmiles(smiles1=smiles1, smiles2=smiles2, method='Tanimoto'))
    print(getStructureSimilarityBySmiles(smiles1=smiles1, smiles2=smiles2, method='Dice'))
    print(getStructureSimilarityBySmiles(smiles1=smiles1, smiles2=smiles2, method='Cosine'))
    print(getStructureSimilarityBySmiles(smiles1=smiles1, smiles2=smiles2, method='Russel'))
