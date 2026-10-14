"""Shared feature dimensions and original pharmacophore descriptors."""
import numpy as np
PHARMACOPHORE_DIM = 16
MOTIF_FEATURE_DIM = 12

def extract_pharmacophore_features(smiles: str) -> np.ndarray:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(16, dtype=np.float32)
        f = np.zeros(16, dtype=np.float32)
        basic_n = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O)]")
        f[0]  = min(len(mol.GetSubstructMatches(basic_n)), 5) / 5.0
        f[1]  = min(rdMolDescriptors.CalcNumAromaticRings(mol), 5) / 5.0
        f[2]  = min(rdMolDescriptors.CalcNumHBD(mol), 10) / 10.0
        f[3]  = min(rdMolDescriptors.CalcNumHBA(mol), 10) / 10.0
        f[4]  = min(rdMolDescriptors.CalcNumRotatableBonds(mol), 15) / 15.0
        f[5]  = Descriptors.MolWt(mol) / 500.0
        f[6]  = (Descriptors.MolLogP(mol) + 5.0) / 15.0
        f[7]  = Descriptors.TPSA(mol) / 150.0
        f[8]  = min(rdMolDescriptors.CalcNumRings(mol), 8) / 8.0
        f[9]  = min(len(mol.GetSubstructMatches(Chem.MolFromSmarts("[r;!c]"))), 5) / 5.0
        f[10] = float(mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1CCN")))
        f[11] = float(mol.HasSubstructMatch(Chem.MolFromSmarts("[C,N]1CCNCC1")))
        f[12] = float(mol.HasSubstructMatch(Chem.MolFromSmarts("c1cnc[nH]1")))
        f[13] = float(
            mol.HasSubstructMatch(Chem.MolFromSmarts("c1cc(O)c(O)cc1")) or
            mol.HasSubstructMatch(Chem.MolFromSmarts("c1ccccc1O"))
        )
        f[14] = min(rdMolDescriptors.CalcNumAtomStereoCenters(mol), 5) / 5.0
        f[15] = rdMolDescriptors.CalcFractionCSP3(mol)
        return np.clip(f, 0.0, 1.0).astype(np.float32)
    except Exception:
        return np.zeros(16, dtype=np.float32)
