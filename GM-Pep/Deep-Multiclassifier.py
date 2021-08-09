import time
from itertools import cycle

import torch
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_curve, \
    auc
from numpy.lib import interp
import matplotlib.pyplot as plt
def feature(f):
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    amino_acids_dict = {}
    seqs = []
    lable_seqs = []

    for n, s in enumerate(amino_acids):
        amino_acids_dict[s] = n

    for n, s in enumerate(SeqIO.parse(f, "fasta")):
        temp_pad = []
        temp_pad1 = []
        temps = []
        for i in range(20):
            temp_pad1.append(0)
        for i in range(100 - len(s)):
            temps.append(temp_pad1)
        for i in range(100 - len(str(s.seq))):
            temp_pad.append(0)
        train_seq = [amino_acids_dict[a.upper()] for a in str(s.seq).upper()] + temp_pad
        seqs.append(train_seq)

        lable_seqs.append([int(s.id[-1])])

    return seqs, lable_seqs


def Load_Data(f, flag):
    seqs, lable_seqs = feature(f)

    seqs = np.array(seqs)
    lable_seqs = np.array(lable_seqs)
    if flag:
        shuffle_idx = np.random.permutation(np.arange(len(seqs)))
        seqs = seqs[shuffle_idx]

        lable_seqs = (torch.from_numpy(lable_seqs[shuffle_idx]).long()) - 1

        temp = torch.zeros(lable_seqs.shape[0], 4)

        lable_seqs = temp.scatter_(1, lable_seqs, 1)

        return torch.from_numpy(seqs), lable_seqs
    else:
        lable_seqs = torch.from_numpy(lable_seqs).long() -1

        temp = torch.zeros(lable_seqs.shape[0], 4)

        lable_seqs = temp.scatter_(1, lable_seqs, 1)

        return torch.from_numpy(seqs), (lable_seqs)



class BiLSTM_Attention(nn.Module):
    def __init__(self, num_layer):
        super(BiLSTM_Attention, self).__init__()
        self.num_layer = num_layer
        self.embedding = nn.Embedding(21, 128)
        self.lstm = nn.LSTM(128, 64, num_layer, bidirectional=True)
        self.out = nn.Linear(256 * int((self.num_layer) / 2), 4)

    def attention_net(self, lstm_output, final_state):
        hidden = final_state.view(-1, 64 * 2, self.num_layer)

        attn_weights = torch.bmm(lstm_output, hidden)
        attn_weights = attn_weights.view(attn_weights.shape[0], -1)
        soft_attn_weights = F.softmax(attn_weights, 1)
        soft_attn_weights = soft_attn_weights.view(soft_attn_weights.shape[0], -1, self.num_layer)

        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights)
        context = context.view(context.shape[0], -1)
        return context, soft_attn_weights

    def forward(self, X ):

        input = self.embedding(X.long().permute(1, 0))

        hidden_state = Variable(torch.zeros(1 * 2* self.num_layer, len(X), 64)).to("cuda:0" if torch.cuda.is_available() else "cpu")
        cell_state = Variable(torch.zeros(1 * 2* self.num_layer, len(X), 64)).to("cuda:0" if torch.cuda.is_available() else "cpu")

        output, (final_hidden_state, final_cell_state) = self.lstm(input, (hidden_state, cell_state))
        output = output.permute(1, 0, 2)
        attn_output, attention = self.attention_net(output, final_hidden_state)

        return torch.nn.functional.log_softmax(self.out(attn_output), 1), attention, output




class Melt_Lstm(torch.nn.Module):
    def __init__(self):
        super(Melt_Lstm, self).__init__()

        self.e1 = nn.Embedding(21, 128)

        self.lstm = nn.LSTM(128, 32, num_layers= 4, bidirectional=True, dropout=0.1)

        self.lstm_a1 = BiLSTM_Attention(4)

        self.conv = nn.Conv1d(100, 50, 1, 1)

        self.conv1 = nn.Conv1d(50, 25, 3, 1)

        self.maxpool1 = nn.MaxPool1d(1, 1)

        self.maxpool = nn.MaxPool1d(3, 1)

        self.fc1 = nn.Linear(9124, 4)

        self.fc2 = nn.Linear(12500, 6024)

        self.fc3 = nn.Linear(6400, 4)

        self.fc4 = nn.Linear(12400, 4)

    def forward(self, x):

        x1 = x.permute(1, 0)

        x_1_ = self.e1(x1.long())

        xa1 = self.conv(x_1_.permute(1, 0, 2))

        xa1 = self.maxpool(xa1)

        xa1 = self.conv1((xa1))

        xa1_1 = self.maxpool1(xa1)

        xa1 = xa1_1.view(xa1_1.shape[0], -1)

        xa_1 = self.conv(x_1_.permute(1, 0, 2))

        xa_1 = self.maxpool(xa_1)

        xa_1 = self.conv1((xa_1))

        xa_11 = self.maxpool1(xa_1)

        xa_1 = xa_11.view(xa_11.shape[0], -1)

        xx = torch.cat((xa_11, xa1_1, xa_11, xa1_1), 1)
        xx1 = xx.view(xx.shape[0], -1)
        xx1 = torch.nn.functional.log_softmax((self.fc4(xx1)), 1)


        h_2A, _ = self.lstm(x_1_)

        h_2 = h_2A.permute(1, 0, 2)

        h_2 = h_2.reshape(h_2.shape[0], -1)

        h_2 = torch.nn.functional.log_softmax((self.fc3(h_2)), 1)


        h_1, attention1, output = self.lstm_a1(x)

        h13 = torch.cat((output, x_1_.permute(1, 0, 2), xx), 2)

        h11 = self.conv(h13)

        h1 = self.maxpool(h11)

        h1 = self.conv1(h1)

        h1 = self.maxpool1(h1)

        h1 = h1.reshape(h1.shape[0], -1)

        out = torch.nn.functional.relu(self.fc2(torch.cat((h1, xa_1), 1)))

        a = torch.nn.functional.log_softmax(self.fc1(torch.cat((out, xa1), 1)), 1)

        a = (a + h_2 + h_1 + xx1) / 4

        return a




def train(seqs, ind_seqs, lable_train, ind_seqs_lable_train):
    r_data = torch.utils.data.TensorDataset(ind_seqs, ind_seqs_lable_train)

    r_dataloder = DataLoader(r_data, shuffle=False, batch_size=128, drop_last=False)

    torch_data = torch.utils.data.TensorDataset(seqs, lable_train)

    dataloder = DataLoader(torch_data, shuffle=False, batch_size=128, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    Cnn_model = Melt_Lstm().to(device)


    loss_func = torch.nn.functional.nll_loss


    optimizer = torch.optim.Adam(Cnn_model.parameters(), lr=1e-3)

    for epoch in range(35):
        loss_it = 0
        total = 0
        train_Acc = 0

        Cnn_model.train()
        for seq, lable_seq in dataloder:
            total += len(seq)

            train_d = seq.to(device).type(torch.cuda.FloatTensor)
            lable_seq = lable_seq.to(device).type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            out = Cnn_model(train_d)

            loss = loss_func(out, torch.argmax(lable_seq, 1))
            loss.backward()
            optimizer.step()

            loss_it += loss.item()

            train_Acc += torch.sum(torch.argmax(out, 1).eq_(torch.argmax(lable_seq, 1))).cpu().tolist()

        score = 0
        print("epoch: {} --> loss: {:5F} --> train_Acc: {:5F} --> val_Acc: {:5F}".format(epoch, (loss_it / total),
                                                                                         (train_Acc / total), score))

    multi_model = Cnn_model.eval()

    r_test_data = 0
    r_test_Acc = 0
    y_pred = []
    true_label = []
    for r_te_seq, r_te_lable_seq in r_dataloder:

        r_te_seq = r_te_seq.to(device).type(torch.cuda.FloatTensor)

        r_te_lable_seq = r_te_lable_seq.to(device).type(torch.cuda.FloatTensor)

        r_test_data += len(r_te_seq)

        r_preds = multi_model(r_te_seq)

        r_preds = torch.softmax(r_preds,1)

        true_label += torch.argmax(r_te_lable_seq, 1).cpu().tolist()

        y_pred += torch.argmax(r_preds, 1).cpu().tolist()

        r_test_Acc += torch.sum(torch.argmax(r_preds, 1).eq_(torch.argmax(r_te_lable_seq, 1))).cpu().tolist()


    print("---------test------------")

    # decode_save_dict = {"Multi_model1": multi_model.state_dict()}
    #
    # torch.save(decode_save_dict, "D:\机器学习\模型\GM-Pep\model_save\\Multi_model_v1.txt")

    return classification_report(true_label, y_pred, digits=4)




def eval(file_predicted):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    amino_acids_dict = {}

    for n, s in enumerate(amino_acids):
        amino_acids_dict[n] = s
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ind_seqs, ind_seqs_lable_train = Load_Data(file_predicted, True)

    r_data = torch.utils.data.TensorDataset(ind_seqs, ind_seqs_lable_train)

    r_dataloder = DataLoader(r_data, shuffle=False, drop_last=False, batch_size=128)

    Cnn_model = Melt_Lstm().to(device)

    save_dict = torch.load("model_save\\Multi_model_v1.txt")

    Cnn_model.load_state_dict(save_dict["Multi_model1"], strict= True)

    Cnn_model = Cnn_model.eval()

    r_test_data = 0
    r_test_Acc = 0
    y_pred = []
    true_label = []
    n = 1
    for r_te_seq, r_te_lable_seq in r_dataloder:
        r_te_seq = r_te_seq.to(device).type(torch.cuda.FloatTensor)
        r_te_lable_seq = r_te_lable_seq.to(device).type(torch.cuda.FloatTensor)
        r_test_data += len(r_te_seq)
        r_preds = Cnn_model(r_te_seq)
        true_label += torch.argmax(r_te_lable_seq, 1).cpu().tolist()
        y_pred += torch.argmax(r_preds, 1).cpu().tolist()
        prob = np.round(torch.softmax(r_preds, 1).detach().cpu().numpy(), 4)
        #print(n)
        #print("prob: ", prob[0])
        #print()
        n += 1
        r_test_Acc += torch.sum(torch.argmax(r_preds, 1).eq_(torch.argmax(r_te_lable_seq, 1))).cpu().tolist()


    #print("---------test------------")
    print(classification_report(true_label, y_pred,digits= 4))

if  __name__ ==  "__main__":
    #  training
    seqs, lable_train = Load_Data("data\\benchmark.txt", True)
    ind_seqs, ind_seqs_lable_train = Load_Data("data\\independent.txt", True)
    r = train(seqs, ind_seqs, lable_train, ind_seqs_lable_train)
    print(r)

    #  predicted
    eval("data\\independent.txt")