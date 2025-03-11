import torch
import torch.nn as nn
import torch.nn.functional as F

from warning_model_utils.pretrain_model import ViT1D


class CPCMinion(nn.Module):

    def __init__(self, inp_dim, n_hidden=None, n_layers=1, name='CPCMinion'):
        super().__init__()

        if n_hidden == None:
            self.n_hidden = inp_dim
        else:
            self.n_hidden = n_hidden

        self.inp_len = inp_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(self.inp_len, self.n_hidden,
                           num_layers=n_layers, batch_first=True)

        self.proj = nn.Linear(self.n_hidden, inp_dim, bias=False)

    def forward(self, input_encoded):
        # input shape bs,ch,seq
        output_rnn, _ = self.rnn(input_encoded)  # output_rnn: bs, seq, n_hidden

        return input_encoded, self.proj(output_rnn)

    def cpc_loss(self, input, target=None, steps_predicted=5, n_false_negatives=9, negatives_from_same_seq_only=False,
                 eval_acc=False):
        # assert (self.num_classes is None)

        input_encoded, output = self.forward(input)  # input_encoded: bs, seq, features; output: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1, input_encoded.size(2))  # for negatives below: -1, features

        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]

        loss = torch.tensor(0, dtype=torch.float32).to(input.device)
        tp_cnt = torch.tensor(0, dtype=torch.int64).to(input.device)

        for i in range(input_encoded.size()[1] - steps_predicted):
            positives = input_encoded[:, i + steps_predicted].unsqueeze(1)  # bs,1,encoder_output_dim
            if (negatives_from_same_seq_only):
                idxs = torch.randint(0, (seq - 1), (bs * n_false_negatives,)).to(input.device)
            else:  # negative from everywhere
                idxs = torch.randint(0, bs * (seq - 1), (bs * n_false_negatives,)).to(input.device)
            idxs_seq = torch.remainder(idxs, seq - 1)  # bs*false_neg
            idxs_seq2 = idxs_seq * (idxs_seq < (i + steps_predicted)).long() + (idxs_seq + 1) * (
                    idxs_seq >= (i + steps_predicted)).long()  # bs*false_neg
            if (negatives_from_same_seq_only):
                idxs_batch = torch.arange(0, bs).repeat_interleave(n_false_negatives).to(input.device)
            else:
                idxs_batch = idxs // (seq - 1)
            idxs2_flat = idxs_batch * seq + idxs_seq2  # for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity

            negatives = input_encoded_flat[idxs2_flat].view(bs, n_false_negatives,
                                                            -1)  # bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives, negatives], dim=1)  # bs,false_neg+1,encoder_output_dim
            preds = torch.sum(output[:, i].unsqueeze(1) * candidates, dim=-1)  # bs,(false_neg+1)
            targs = torch.zeros(bs, dtype=torch.int64).to(input.device)

            if (eval_acc):
                preds_argmax = torch.argmax(preds, dim=-1)
                tp_cnt += torch.sum(preds_argmax == targs)

            loss += F.cross_entropy(preds, targs)
        if (eval_acc):
            return loss, tp_cnt.float() / bs / (input_encoded.size()[1] - steps_predicted)
        else:
            return loss


class cpcPretrainModel(nn.Module):
    def __init__(self,
                 encoder="vit", num_classes=5):

        super().__init__()
        # init frontend
        self.encoder_name = encoder
        if "vit" in encoder:
            self.frontend = ViT1D()
            dim = 256
        else:
            print("wrong encoder name : {}".format(self.encoder_name))
            pass

        self.cpc_head = CPCMinion(inp_dim=dim, n_hidden=256, n_layers=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.multi_label_classifier = nn.Linear(dim, num_classes)

    def forward(self, x):
        device = x.device


        if self.encoder_name == "vit":
            h = self.frontend(x)
        else:

            h = self.frontend.forward_features(x)
            h = h.squeeze(2)
            h = h.permute(0, 2, 1)
        return self.cpc_head.cpc_loss(h)

    def classify(self, signal):
        if self.encoder_name == "vit":
            encoded_tokens = self.frontend(signal)
        else:

            encoded_tokens = self.frontend.forward_features(signal)
            encoded_tokens = encoded_tokens.squeeze(2)
            encoded_tokens = encoded_tokens.permute(0, 2, 1)
            # print(encoded_tokens.shape)

        latentFeature = self.avgpool(encoded_tokens.permute(0, 2, 1))
        latentFeature = latentFeature.view(latentFeature.size(0), -1)
        out = self.multi_label_classifier(latentFeature)
        return out