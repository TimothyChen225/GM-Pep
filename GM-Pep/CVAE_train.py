from CVAE.Encode import Encoder
from CVAE.Decode import Decoder
import torch
from utils.CVAE_utils import Load_Data, batch_Decode_sample
import numpy as np
import argparse


def loss_func(recons_out, recons_int, mu, logvar):
    # print(mu.shape)
    BCE = torch.nn.functional.binary_cross_entropy(recons_out, recons_int, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, KLD

def run(dataloader, hidden_size, epochs):
    encoder = Encoder()
    decoder = Decoder(hidden_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    optimizerD = torch.optim.Adam(decoder.parameters(), lr=0.001)
    optimizerE = torch.optim.Adam(encoder.parameters(), lr=0.001)


    ll = []
    for epoch in range(epochs):
        loss_l = []
        KLD_l = []
        reconstruction = []
        for i, (data, label) in enumerate(dataloader):
            data = data.type(torch.cuda.FloatTensor)  # [batch_size, 1 , 47, 41]

            label = label.long().to(device) - 2  # [batch_size, 1]

            temp = torch.zeros(label.shape[0], 3, device=device)  # [batch_size, 10]

            re_temp = temp.scatter_(1, label, 1).unsqueeze(1)  # [batch_size, 10]

            re_temp1 = re_temp.expand([re_temp.shape[0], 100, re_temp.shape[2]]).unsqueeze(1)  # [batch_size, 1, 47, 10]
            #print(re_temp1.shape)
            recons_int = data.squeeze(1)  # [batch_size, 47, 41]

            optimizerD.zero_grad()
            optimizerE.zero_grad()

            z, mu, logvar = encoder(data, re_temp1)

            recons_out = decoder(z, re_temp)  # # [batch_size, 47, 41]

            if epoch == 5999:
                n = batch_Decode_sample(recons_out, recons_int, epoch)
                reconstruction.extend(n)

            loss, KLD = loss_func(recons_out, recons_int, mu, logvar)

            loss.backward()

            optimizerD.step()
            optimizerE.step()

            loss_l.append(loss.item())
            KLD_l.append(KLD.item())
        if epoch == 5999:
            ll.append(np.mean(reconstruction))
            #ll.append()
            print("reconst_ACC: {} ==> reconst_num: {}".format(np.mean(reconstruction), np.sum(reconstruction)))
        print("第{}epoch: loss: {} || KLD: {}".format(epoch, np.mean(loss_l), np.mean(KLD_l)))

    # encode_save_dict = {"Encode_model": encoder.state_dict()}
    # decode_save_dict = {"Decode_model": decoder.state_dict()}
    # torch.save(encode_save_dict, "D:\机器学习\模型\GM-Pep\model_save\\tem_Encode_modelMuti_v3.txt")
    # torch.save(decode_save_dict, "D:\机器学习\模型\GM-Pep\model_save\\tem_Decode_modelMuti_v3.txt")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=" Load  args and train a encoder_decoder model")
    parser.add_argument("-f", "--filname", help="filname of peptids", default="data\\TC_1.txt")
    parser.add_argument("-hs", "--hidden_size", default=1000)
    parser.add_argument("-bs", "--batch_size", default=300)
    parser.add_argument("-e", "--epochs", default= 6000)

    args = parser.parse_args()

    dataloader = Load_Data(args.filname, args.batch_size)

    run(dataloader, args.hidden_size, args.epochs)

