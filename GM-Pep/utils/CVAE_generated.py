import torch

from collections import Counter
from CVAE.Decode import Decoder

import numpy as np

from utils.CVAE_utils import decode_Sample


def generated_Peptides(hidden_size, num_Class, filname, model_filname, write_filname):
    ap = []
    ap1 = []
    filname = filname
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    save_dic = torch.load(model_filname)
    D = Decoder(hidden_size)
    D = D.to(device)
    D.load_state_dict(save_dic["Decode_model"], strict=True)
    D = D.eval()

    type_index = [0] * num_Class + [1] + [0] * (3 - num_Class - 1)

    type_index = torch.from_numpy(np.array(type_index)).long()
    type_lable = type_index.expand(10000, 3).unsqueeze(1).float().to(device)

    z = torch.randn(10000, 1, 350).type(torch.cuda.FloatTensor).to(device)

    w = open(write_filname, "ta", encoding="utf-8")

    with open(filname) as f:
        for i in f:
            i = i.strip()
            if i[0] != ">":
                ap1.append(i)

    decode_data = D(z, type_lable)

    for n, re_out in enumerate(decode_data):
        recon_pep = decode_Sample(re_out, None, "test")

        for indx, aa in enumerate(recon_pep):

            if recon_pep[indx:] == "X" * (len(recon_pep) - indx):

                if recon_pep[:indx] != None and "X" not in recon_pep[:indx] and "x" not in recon_pep[
                                                                                           :indx] and recon_pep[
                                                                                                      :indx] not in ap1:
                    ap.append(recon_pep[:indx])

                    break
    dulp = []
    for i in set(ap):

        if i not in ap1:

            if len(i) >= 5:
                dulp.append(i)

    re_dulp = Counter(dulp)
    for i in dulp:
        if re_dulp[i] < 2:
            w.write(">{}\n".format(num_Class + 2))
            w.write("{}\n".format(i))


if __name__ == "__main__":
    # provided TC_2 dataset path
    filname = "data\\total.txt"
    # provided encoder model saved path
    model_filname = "model_save\\tem_Decode_modelMulti_v2.txt"
    # provided relevant sequences writing path
    write_filname = "data\data\\temp.fa"
    # provided the number of wanted class sequences
    # 0 represent anti-fungal, 1 represent anti-hypertensive, 2 represent anti-bacterial
    class_num = 0
    generated_Peptides(1000, class_num, filname, model_filname, write_filname)
