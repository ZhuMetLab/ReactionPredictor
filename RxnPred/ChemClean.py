from rdkit import Chem
from rdkit.Chem import MolStandardize, rdMolDescriptors


class MolCleaner(object):
    def __init__(self):
        self.normizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def _cleanMol(self, mol):
        mol = self.normizer.normalize(mol)
        mol = self.lfc.choose(mol)
        mol = self.uc.uncharge(mol)
        return mol

    def cleanSmiles(self, smiles, isomeric=True, kekule=True):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = self._cleanMol(mol)
            smiles = Chem.MolToSmiles(mol, isomericSmiles=isomeric, kekuleSmiles=kekule)
            return smiles
        else:
            return None

    def smiles2inchikey(self, smiles, is_clean=True):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if is_clean:
                mol = self._cleanMol(mol)
            return Chem.inchi.MolToInchiKey(mol)
        else:
            return None

    def inchi2smiles(self, inchi, is_clean=True, isomeric=True, kekule=True):
        mol = Chem.MolFromInchi(inchi)
        if mol:
            if is_clean:
                mol = self._cleanMol(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=isomeric, kekuleSmiles=kekule)
        else:
            return None

    def inchi2inchikey(self, inchi, is_clean=True):
        mol = Chem.MolFromInchi(inchi)
        if mol:
            if is_clean:
                mol = self._cleanMol(mol)
            return Chem.inchi.MolToInchiKey(mol)
        else:
            return None

    def smiles2formula(self, smiles, is_clean=True):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            if is_clean:
                mol = self._cleanMol(mol)
            formula = rdMolDescriptors.CalcMolFormula(mol)
            return formula
        else:
            return None
