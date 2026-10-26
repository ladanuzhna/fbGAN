import numpy as np
import pandas as pd

# Adapted from https://github.com/niranjangavade98/DNA-to-Protein-sequence
# and https://github.com/av1659/fbgan/blob/master/utils/bio_utils.py

DNA_protein_MAP = {
    'ATA': 'I',
    'ATC': 'I',
    'ATT': 'I',
    'ATG': 'M',
    'ACA': 'T',
    'ACC': 'T',
    'ACG': 'T',
    'ACT': 'T',
    'AAC': 'N',
    'AAT': 'N',
    'AAA': 'K',
    'AAG': 'K',
    'AGC': 'S',
    'AGT': 'S',
    'AGA': 'R',
    'AGG': 'R',
    'CTA': 'L',
    'CTC': 'L',
    'CTG': 'L',
    'CTT': 'L',
    'CCA': 'P',
    'CCC': 'P',
    'CCG': 'P',
    'CCT': 'P',
    'CAC': 'H',
    'CAT': 'H',
    'CAA': 'Q',
    'CAG': 'Q',
    'CGA': 'R',
    'CGC': 'R',
    'CGG': 'R',
    'CGT': 'R',
    'GTA': 'V',
    'GTC': 'V',
    'GTG': 'V',
    'GTT': 'V',
    'GCA': 'A',
    'GCC': 'A',
    'GCG': 'A',
    'GCT': 'A',
    'GAC': 'D',
    'GAT': 'D',
    'GAA': 'E',
    'GAG': 'E',
    'GGA': 'G',
    'GGC': 'G',
    'GGG': 'G',
    'GGT': 'G',
    'TCA': 'S',
    'TCC': 'S',
    'TCG': 'S',
    'TCT': 'S',
    'TTC': 'F',
    'TTT': 'F',
    'TTA': 'L',
    'TTG': 'L',
    'TAC': 'Y',
    'TAT': 'Y',
    'TAA': '',
    'TAG': '',
    'TGC': 'C',
    'TGT': 'C',
    'TGA': '',
    'TGG': 'W',
}

protein_DNA_MAP = {v: k for k, v in DNA_protein_MAP.items()}
protein_DNA_MAP.pop('', None)


def parse(sequences):
    if type(sequences) == str:
        parsed = np.array([a for a in sequences])
        return parsed

    parse = lambda seq: np.array([a for a in seq])
    parsed = pd.DataFrame(sequences).iloc[:, 0].apply(parse).to_numpy()

    return parsed


def protein_to_DNA(protein_sequences):
    """Encode proteins with one codon per residue and start/stop markers."""
    if isinstance(protein_sequences, str):
        protein_sequences = [protein_sequences]
    sequences = np.asarray(protein_sequences).reshape(-1)
    encoded = ['ATG' + ''.join(protein_DNA_MAP[aa] for aa in seq) + 'TAG' for seq in sequences]
    return np.asarray(encoded).reshape(-1, 1)


def translate(seq):
    """Translate a DNA string or one-element row, stopping at padding or a stop codon."""
    if not isinstance(seq, str):
        seq = seq[0]
    seq = seq.split('P')[0]
    protein = []
    for i in range(0, len(seq) - 2, 3):
        amino_acid = DNA_protein_MAP.get(seq[i:i + 3])
        if amino_acid == '':
            break
        if amino_acid is not None:
            protein.append(amino_acid)
    return ''.join(protein)


def DNA_to_protein(sequences):
    if isinstance(sequences, str):
        sequences = [sequences]
    return [translate(seq) for seq in sequences]
