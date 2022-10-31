
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.feature import Feature



def Load_Data(filname, batch_size):
    seq, labels = Feature(filname)
    seq = torch.from_numpy(np.array(seq))
    labels = torch.from_numpy(np.array(labels))

    seq = seq.unsqueeze(1)

    seqs = torch.utils.data.TensorDataset(seq, labels)
    train_dataloder = DataLoader(seqs, shuffle= True, batch_size= batch_size, drop_last= True)
    return train_dataloder

def batch_Decode_sample(batch_recons_out, batch_recons_int, epoch):
    temp = []
    for batch_id, recons_out in enumerate(batch_recons_out):
        if batch_recons_int != None:
            n = decode_Sample(recons_out, batch_recons_int[batch_id], epoch)
            temp.append(n)
        else:
            decode_Sample(recons_out.reshape(100, 21), batch_recons_int, epoch)
    return temp

def decode_Sample(recons_out, recons_int, epoch):

    if epoch == "test" and recons_int == None:
        recons_out = recons_out.clamp(min=0.10).cpu()
        #recons_int = recons_int.cpu()
        re, index = torch.max(recons_out, dim=1)
        #print(re)
        recons_like = torch.zeros(recons_out.shape)

        for n, i in enumerate(index):
            if re[n] > 0.10:
                recons_like[n, i] = 1
            else:
                recons_like[n,0] = 1
        recon_pep, pep = show_Peptids(recons_like, recons_int)

        return recon_pep
    else:

            recons_out = recons_out.clamp(min=0.1).cpu()
            recons_int = recons_int.cpu()
            re, index = torch.max(recons_out, dim=1)
            # print(re)
            recons_like = torch.zeros(recons_out.shape)

            for n, i in enumerate(index):
                if re[n] > 0.100:
                    recons_like[n, i] = 1
                else:
                    recons_like[n, 0] = 1
            recon_pep, pep = show_Peptids(recons_like, recons_int)
            print("recon_pep : {} || length: {}".format(recon_pep, len(recon_pep)))
            print("pep : {} || length: {}".format(pep, len(pep)))
            if recon_pep == pep:
                print("---------------------")
                return 1

            return 0

def show_Peptids(recons_like, recons_int):
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}

    recon_pep = ""
    pep = ""

    for n, i in enumerate(amino_acids):
        amino_acids_dict[n] = i

    for row in range(recons_like.shape[0]):
        for col in range(recons_like.shape[1]):
            if int(recons_like[row, col]) == 1:
                recon_pep += amino_acids_dict[int(col)]
    if recons_int != None:
        for row in range(recons_int.shape[0]):
            if max(recons_int[row]) == 0:
                pep += "X"
            for col in range(recons_int.shape[1]):
                if int(recons_int[row, col]) == 1:
                    pep += amino_acids_dict[int(col)]

    return recon_pep, pep

def Sampler(batch_size):

    z = torch.randn(batch_size, 350)
    return z