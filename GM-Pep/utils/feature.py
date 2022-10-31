from Bio import SeqIO


def add(x, i):
    x_copy = x.copy()
    x_copy[i] = 1
    return x_copy


def Feature(filname):
    seq = []  # a list containing total matrix of peptides
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}
    pad = []
    labels = []

    # a list padded zero in peptide in order to keep same dimension with lengths of max
    for i in range(len(amino_acids)):
        pad.append(0)

    # a dictionary mapping amino acid to number
    for n, s in enumerate(amino_acids):
        amino_acids_dict[s] = n

    # transforming  peptides to a two dimension matrix
    for n, s in enumerate(SeqIO.parse(filname, "fasta")):

        label = int(s.id)
        labels.append([label])
        seq.append([])
        temp_pad = []
        temp1 = []

        for i in range(100 - len(str(s.seq))):
            temp_pad.append(pad)

        for j, i in enumerate(str(s.seq)):
            temp1.append([])

            temp1[j] = add(pad, amino_acids_dict[i])

        temp1 += temp_pad

        seq[n] = temp1
    return seq, labels

# Feature()
