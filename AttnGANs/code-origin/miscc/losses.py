import torch
import torch.nn as nn

from torch.autograd import Variable
from torch import autograd

from torch.nn import functional as F
from torchvision.models.inception import inception_v3
from scipy.stats import entropy

import pickle
import numpy as np
from miscc.config import cfg

from GlobalAttention import func_attention

# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """
        Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(global_image_feature, sentence_feature, labels, captions_ids, batch_size, eps=1e-8):
    # ### Mask mis-match samples  ###
    # that come from the same class as the real sample ###
    masks = []
    if captions_ids is not None:
        for i in range(batch_size):
            mask = (captions_ids == captions_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    # --> seq_len x batch_size x embedding_dim
    if global_image_feature.dim() == 2:
        global_image_feature = global_image_feature.unsqueeze(0)
        sentence_feature = sentence_feature.unsqueeze(0)

    # global_image_feature_norm / sentence_feature_norm: seq_len x batch_size x 1
    global_image_feature_norm = torch.norm(global_image_feature, 2, dim=2, keepdim=True)
    sentence_feature_norm = torch.norm(sentence_feature, 2, dim=2, keepdim=True)
    # scores* / norm*: seq_len x batch_size x batch_size
    scores1 = torch.bmm(global_image_feature, sentence_feature.transpose(1, 2))
    norm1 = torch.bmm(global_image_feature_norm, sentence_feature_norm.transpose(1, 2))
    scores1 = scores1 / norm1.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3

    # --> batch_size x batch_size
    scores1 = scores1.squeeze()
    if captions_ids is not None:
        scores1.data.masked_fill_(masks, -float('inf'))
    scores2 = scores1.transpose(0, 1)
    if labels is not None:
        s_loss_1 = nn.CrossEntropyLoss()(scores1, labels)
        s_loss_2 = nn.CrossEntropyLoss()(scores2, labels)
    else:
        s_loss_1, s_loss_2 = None, None
    '''
    print('Eq.(7):', norm1)
    print('Eq.(9):', scores1)
    print('Eq.(12):', scores2)
    print('Eq.(10)[s]:', s_loss_1)
    print('Eq.(14)[s]:', s_loss_2)
    '''
    return s_loss_1, s_loss_2


def words_loss(local_image_features, words_features, labels, captions_lens, captions_ids, batch_size, epoch, mode='train'):
    """
        words_features(query): batch x embedding_dim x seq_len
        local_image_features(context): batch x embedding_dim x 8 x 8
    """
    masks = []
    att_maps = []
    similarities1 = []
    similarity1 = 0
    similarity2 = 0
    captions_lens = captions_lens.data.tolist()
    for i in range(batch_size):
        if captions_ids is not None:
            mask = (captions_ids == captions_ids[i]).astype(np.uint8)
            mask[i] = 0
            masks.append(mask.reshape((1, -1)))

        # Get the i-th text description
        words_num = captions_lens[i]
        # -> 1 x embedding_dim x words_num
        word = words_features[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x embedding_dim x words_num
        word = word.repeat(batch_size, 1, 1).cuda()
        # batch x embedding_dim x 8*8
        context = local_image_features
        """
            word(query): batch x embedding_dim x words_num
            context: batch x embedding_dim x 8 x 8
            weiContext: batch x embedding_dim x words_num
            attn: batch x words_num x 8 x 8
        """
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x embedding_dim
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x embedding_dim
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities1(i, j): the similarity between the i-th image and the j-th text description
        similarities1.append(row_sim)

    # batch_size x batch_size
    similarities1 = torch.cat(similarities1, 1)
    if captions_ids is not None:
        masks = np.concatenate(masks, 0)
        # masks: batch_size x batch_size
        masks = torch.ByteTensor(masks)
        if cfg.CUDA:
            masks = masks.cuda()

    similarities1 = similarities1 * cfg.TRAIN.SMOOTH.GAMMA3

    if captions_ids is not None:
        similarities1.data.masked_fill_(masks, -float('inf'))
    similarities2 = similarities1.transpose(0,1)
    if labels is not None:
        w_loss_1 = nn.CrossEntropyLoss()(similarities1, labels)
        w_loss_2 = nn.CrossEntropyLoss()(similarities2, labels)
    else:
        w_loss_1, w_loss_2 = None, None
    '''
    print('Eq.(3):', weiContext)
    print('Eq.(4):', attn)
    print('Eq.(5):', att_maps)
    print('Eq.(6):', row_sim)
    print('Eq.(8):', similarities1)
    print('Eq.(11):', similarities2)
    print('Eq.(10)[w]:', w_loss_1)
    print('Eq.(13)[w]:', w_loss_2)
    '''
    if mode == 'test':
        with open('../output/%s_DAMSM/Numeric/%s_numeric_%04d.pickle'%(cfg.DATASET_NAME, cfg.DATASET_NAME, epoch), 'wb') as f:
            pickle.dump([weiContext, attn, att_maps, row_sim, similarities1, similarities2, w_loss_1, w_loss_2], f)
    return w_loss_1, w_loss_2, att_maps

# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions, real_labels, fake_labels):
    # Forward
    real_features = netD(real_imgs)
    fake_features = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCEWithLogitsLoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCEWithLogitsLoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCEWithLogitsLoss()(cond_wrong_logits, fake_labels[1:batch_size])

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCEWithLogitsLoss()(real_logits, real_labels)
        fake_errD = nn.BCEWithLogitsLoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD)/2 + (fake_errD + cond_fake_errD + cond_wrong_errD))/3
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD)/2
    return errD


def generator_loss(netsD, image_encoder, fake_imgs, real_labels, words_featuress, sentence_feature, match_labels, captions_lens, captions_ids, count):
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(len(netsD)):
        features = netsD[i](fake_imgs[i])
        cond_logits = netsD[i].COND_DNET(features, sentence_feature)
        cond_errG = nn.BCEWithLogitsLoss()(cond_logits, real_labels)
        if netsD[i].UNCOND_DNET is  not None:
            logits = netsD[i].UNCOND_DNET(features)
            errG = nn.BCEWithLogitsLoss()(logits, real_labels)
            g_loss = errG + cond_errG
        else:
            g_loss = cond_errG
        errG_total += g_loss
        # err_img = errG_total.data.item()
        logs += 'g_loss%d: %.7f ' % (i, g_loss.data.item())

        # Ranking loss
        if i == (len(netsD) - 1):
            # local_image_features: batch_size x embedding_dim x 8 x 8
            # sent_code: batch_size x embedding_dim
            local_features, global_image_feature = image_encoder(fake_imgs[i])
            w_loss0, w_loss1, _ = words_loss(local_features, words_featuress, match_labels, captions_lens, captions_ids, batch_size, count)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_words = err_words + w_loss.data.item()

            s_loss0, s_loss1 = sent_loss(global_image_feature, sentence_feature, match_labels, captions_ids, batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
            # err_sent = err_sent + s_loss.data.item()

            errG_total += w_loss + s_loss
            logs += '\nw_loss: %.7f s_loss: %.7f ' % (w_loss.data.item(), s_loss.data.item())
    return errG_total, logs

##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD
