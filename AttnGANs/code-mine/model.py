import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from GlobalAttention import GlobalAttentionGeneral as ATT_NET


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)

def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2), nn.BatchNorm2d(out_planes * 2), GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2), nn.BatchNorm2d(channel_num * 2), GLU(),conv3x3(channel_num, channel_num), nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of local_image_features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: If non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True, dropout=self.drop_prob, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden, self.nlayers, batch_first=True, dropout=self.drop_prob, bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, batchsize):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions, batchsize, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions, batchsize, self.nhidden).zero_()))
        else:
            return  Variable(weight.new(self.nlayers * self.num_directions, batchsize, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, captions_vector, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        emb = self.drop(captions_vector)
        #
        # Returns: a PackedSequence object
        cap_lens = np.array(cap_lens.data.tolist())
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output local_image_features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_ENCODER, self).__init__()
        if cfg.TRAIN.FLAG:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = 300  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(model)

        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_local_image_features = conv1x1(768, self.embedding_dim)
        self.emb_global_image_feature = nn.Linear(2048, self.embedding_dim)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_local_image_features.weight.data.uniform_(-initrange, initrange)
        self.emb_global_image_feature.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        local_image_features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region local_image_features
        local_image_features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global_image_feature
        global_image_feature = self.emb_global_image_feature(x)
        # 512
        if local_image_features is not None:
            local_image_features = self.emb_local_image_features(local_image_features)
        return local_image_features, global_image_feature


# ############## G networks ###################
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, gf_dim, condition_dim):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = gf_dim
        self.in_dim = cfg.GAN.Z_DIM + condition_dim  # cfg.TEXT.EMBEDDING_DIM

        self.define_module()

    def define_module(self):
        nz, gf_dim = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(nn.Linear(nz, gf_dim * 4 * 4 * 2, bias=False), nn.BatchNorm1d(gf_dim * 4 * 4 * 2), GLU())

        self.upsample1 = upBlock(gf_dim, gf_dim // 2)
        self.upsample2 = upBlock(gf_dim // 2, gf_dim // 4)
        self.upsample3 = upBlock(gf_dim // 4, gf_dim // 8)
        self.upsample4 = upBlock(gf_dim // 8, gf_dim // 16)

    def forward(self, z_code, c_code):
        """
        :param z_code: batch x cfg.GAN.Z_DIM
        :param c_code: batch x cfg.TEXT.EMBEDDING_DIM
        :return: batch x gf_dim/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size gf_dim x 4 x 4
        out_code = self.fc(c_z_code)
        out_code4 = out_code.view(-1, self.gf_dim, 4, 4)
        # state size gf_dim/3 x 8 x 8
        out_code8 = self.upsample1(out_code4)
        # state size gf_dim/4 x 16 x 16
        out_code16 = self.upsample2(out_code8)
        # state size gf_dim/8 x 32 x 32
        out_code32 = self.upsample3(out_code16)
        # state size gf_dim/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, gf_dim, embedding_dim, condition_dim):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = gf_dim
        self.ef_dim = embedding_dim
        self.cf_dim = condition_dim
        self.num_residual = cfg.GAN.R_NUM
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.GAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        gf_dim = self.gf_dim
        self.att = ATT_NET(gf_dim, self.ef_dim)
        self.residual = self._make_layer(ResBlock, gf_dim * 2)
        self.upsample = upBlock(gf_dim * 2, gf_dim)

    def forward(self, h_code, c_code, word_embs, mask):
        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        self.att.applyMask(mask)
        c_code, att = self.att(h_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        out_code = self.residual(h_c_code)

        # state size gf_dim/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, gf_dim):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = gf_dim
        self.img = nn.Sequential(conv3x3(gf_dim, 3), nn.Tanh())

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        gf_dim = cfg.GAN.GF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        condition_dim = cfg.GAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(gf_dim * 16, condition_dim)
            self.img_net1 = GET_IMAGE_G(gf_dim)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(gf_dim, embedding_dim, condition_dim)
            self.img_net2 = GET_IMAGE_G(gf_dim)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(gf_dim, embedding_dim, condition_dim)
            self.img_net3 = GET_IMAGE_G(gf_dim)
        if cfg.TREE.BRANCH_NUM > 3:
            self.h_net4 = NEXT_STAGE_G(gf_dim, embedding_dim, condition_dim)
            self.img_net4 = GET_IMAGE_G(gf_dim)
        if cfg.TREE.BRANCH_NUM > 4:
            self.h_net5 = NEXT_STAGE_G(gf_dim, embedding_dim, condition_dim)
            self.img_net5 = GET_IMAGE_G(gf_dim)

    def forward(self, z_code, sent_emb, word_embs, mask):
        """
            :param z_code: batch x cfg.GAN.Z_DIM
            :param sent_emb: batch x cfg.TEXT.EMBEDDING_DIM
            :param word_embs: batch x cdf x seq_len
            :param mask: batch x seq_len
            :return:
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2, att1 = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4, att3 = self.h_net4(h_code3, c_code, word_embs, mask)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)
            if att3 is not None:
                att_maps.append(att3)
        if cfg.TREE.BRANCH_NUM > 4:
            h_code5, att4 = self.h_net5(h_code4, c_code, word_embs, mask)
            fake_img5 = self.img_net5(h_code5)
            fake_imgs.append(fake_img5)
            if att4 is not None:
                att_maps.append(att4)

        return fake_imgs, att_maps, mu, logvar

# ############## D networks ##########################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(df_dim):
    encode_img = nn.Sequential(
        # --> state size. df_dim x in_size/2 x in_size/2
        nn.Conv2d(3, df_dim, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2df_dim x x in_size/4 x in_size/4
        nn.Conv2d(df_dim, df_dim * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(df_dim * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4df_dim x in_size/8 x in_size/8
        nn.Conv2d(df_dim * 2, df_dim * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(df_dim * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8df_dim x in_size/16 x in_size/16
        nn.Conv2d(df_dim * 4, df_dim * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(df_dim * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


class D_GET_LOGITS(nn.Module):
    def __init__(self, df_dim, embedding_dim, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = embedding_dim
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(df_dim * 8 + embedding_dim, df_dim * 8)

        self.outlogits = nn.Sequential(nn.Conv2d(df_dim * 8, 1, kernel_size=4, stride=4), nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (gf_dim+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)
            # state size gf_dim x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET64, self).__init__()
        df_dim = cfg.GAN.DF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(df_dim)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=True)

    def forward(self, x_var):
        x_code4 = self.img_code_s16(x_var)  # 4 x 4 x 8df
        return x_code4


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET128, self).__init__()
        df_dim = cfg.GAN.DF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(df_dim)
        self.img_code_s32 = downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        #
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=True)

    def forward(self, x_var):
        x_code8 = self.img_code_s16(x_var)   # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)   # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        return x_code4


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET256, self).__init__()
        df_dim = cfg.GAN.DF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(df_dim)
        self.img_code_s32 = downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s64 = downBlock(df_dim * 16, df_dim * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(df_dim * 32, df_dim * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=True)

    def forward(self, x_var):
        x_code16 = self.img_code_s16(x_var)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        return x_code4

# For 512 x 512 images
class D_NET512(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET512, self).__init__()
        df_dim = cfg.GAN.DF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(df_dim)
        self.img_code_s32 = downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s64 = downBlock(df_dim * 16, df_dim * 32)
        self.img_code_s128 = downBlock(df_dim * 32, df_dim * 64)
        self.img_code_s128_1 = Block3x3_leakRelu(df_dim * 64, df_dim * 32)
        self.img_code_s128_2 = Block3x3_leakRelu(df_dim * 32, df_dim * 16)
        self.img_code_s128_3 = Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=True)

    def forward(self, x_var):
        x_code32 = self.img_code_s16(x_var)
        x_code16 = self.img_code_s32(x_code32)
        x_code8 = self.img_code_s64(x_code16)
        x_code4 = self.img_code_s128(x_code8)
        x_code4 = self.img_code_s128_1(x_code4)
        x_code4 = self.img_code_s128_2(x_code4)
        x_code4 = self.img_code_s128_3(x_code4)
        return x_code4

# For 1024 x 1024 images
class D_NET1024(nn.Module):
    def __init__(self, b_jcu=True):
        super(D_NET1024, self).__init__()
        df_dim = cfg.GAN.DF_DIM
        embedding_dim = cfg.TEXT.EMBEDDING_DIM
        self.img_code_s16 = encode_image_by_16times(df_dim)
        self.img_code_s32 = downBlock(df_dim * 8, df_dim * 16)
        self.img_code_s64 = downBlock(df_dim * 16, df_dim * 32)
        self.img_code_s128 = downBlock(df_dim * 32, df_dim * 64)
        self.img_code_s256 = downBlock(df_dim * 64, df_dim * 128)
        self.img_code_s256_1 = Block3x3_leakRelu(df_dim * 128, df_dim * 64)
        self.img_code_s256_2 = Block3x3_leakRelu(df_dim * 64, df_dim * 32)
        self.img_code_s256_3 = Block3x3_leakRelu(df_dim * 32, df_dim * 16)
        self.img_code_s256_4 = Block3x3_leakRelu(df_dim * 16, df_dim * 8)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=False)
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(df_dim, embedding_dim, bcondition=True)

    def forward(self, x_var):
        x_code64 = self.img_code_s16(x_var)
        x_code32 = self.img_code_s32(x_code64)
        x_code16 = self.img_code_s64(x_code32)
        x_code8 = self.img_code_s128(x_code16)
        x_code4 = self.img_code_s256(x_code8)
        x_code4 = self.img_code_s256_1(x_code4)
        x_code4 = self.img_code_s256_2(x_code4)
        x_code4 = self.img_code_s256_3(x_code4)
        x_code4 = self.img_code_s256_4(x_code4)
        return x_code4
