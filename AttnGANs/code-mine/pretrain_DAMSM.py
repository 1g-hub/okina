from __future__ import print_function

import warnings
warnings.simplefilter("ignore", UserWarning)

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import shutil
import glob
import time
import random
import pprint
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

sys.path.append(os.pardir)
dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='cfg/DAMSM/concept-net.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def train(dataloader, cnn_model, rnn_model, batch_size, labels, optimizer, epoch, ixtoword, image_dir):

    UPDATE_INTERVAL = len(dataloader)

    cnn_model.train()
    rnn_model.train()

    s_total_loss_1 = 0
    s_total_loss_2 = 0
    w_total_loss_1 = 0
    w_total_loss_2 = 0
    count = (epoch + 1) * len(dataloader)

    for step, data in enumerate(dataloader, 1):
        # print('step', step)
        rnn_model.zero_grad()
        cnn_model.zero_grad()

        imgs, captions, cap_lens, class_ids, keys, captions_vector = prepare_data(data)

        # local_image_features: batch_size x embedding_dim x 17 x 17
        # global_image_feature: batch_size x embedding_dim
        local_image_features, global_image_feature = cnn_model(imgs[-1])
        # --> batch_size x embedding_dim x 17*17
        embedding_dim, att_sze = local_image_features.size(1), local_image_features.size(2)
        # local_image_features = local_image_features.view(batch_size, embedding_dim, -1)

        hidden = rnn_model.init_hidden(batch_size)
        # words_features: batch_size x embedding_dim x seq_len
        # sentence_feature: batch_size x embedding_dim
        words_features, sentence_feature = rnn_model(captions, cap_lens, hidden, captions_vector)

        w_loss_1, w_loss_2, attn_maps = words_loss(local_image_features, words_features, labels, cap_lens, class_ids, batch_size, epoch, mode='train')
        w_total_loss_1 += w_loss_1.data
        w_total_loss_2 += w_loss_2.data
        w_loss = w_loss_1 + w_loss_2

        s_loss_1, s_loss_2 = sent_loss(global_image_feature, sentence_feature, labels, class_ids, batch_size)
        s_total_loss_1 += s_loss_1.data
        s_total_loss_2 += s_loss_2.data
        s_loss = s_loss_1 + s_loss_2
        #
        loss = w_loss + s_loss
        loss.backward()

        #
        # `clip_grad_norm` helps prevent
        # the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), cfg.TRAIN.RNN_GRAD_CLIP)
        optimizer.step()

    s_cur_loss_1 = s_total_loss_1.item() / step
    s_cur_loss_2 = s_total_loss_2.item() / step

    w_cur_loss_1 = w_total_loss_1.item() / step
    w_cur_loss_2 = w_total_loss_2.item() / step

    print('| epoch {:4d}/{:4d} | training   loss | ''s_loss_1:{} | ''s_loss_2:{} | ''w_loss_1:{} | ''w_loss_2:{} |'
             .format(epoch, cfg.TRAIN.MAX_EPOCH, s_cur_loss_1, s_cur_loss_2, w_cur_loss_1, w_cur_loss_2))

    # attention Maps
    img_set, _ = build_super_images(imgs[-1].cpu(), captions, ixtoword, attn_maps, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/train_attention_maps_%04d.jpg' % (image_dir, epoch)
        im.save(fullpath)

    training = [w_cur_loss_1, w_cur_loss_2, s_cur_loss_1, s_cur_loss_2]

    return training

def evaluate(dataloader, cnn_model, rnn_model, batch_size, epoch, ixtoword, image_dir):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss_1 = 0
    s_total_loss_2 = 0
    w_total_loss_1 = 0
    w_total_loss_2 = 0
    for step, data in enumerate(dataloader, 1):
        real_imgs, captions, cap_lens, class_ids, keys, captions_vector = prepare_data(data)

        local_image_features, global_image_feature = cnn_model(real_imgs[-1])
        # embedding_dim = local_image_features.size(1)
        embedding_dim, att_sze = local_image_features.size(1), local_image_features.size(2)
        # local_image_features = local_image_features.view(batch_size, embedding_dim, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_features, sentence_feature = rnn_model(captions, cap_lens, hidden, captions_vector)

        w_loss_1, w_loss_2, attn = words_loss(local_image_features, words_features, labels, cap_lens, class_ids, batch_size, epoch, mode='eval')
        w_total_loss_1 += w_loss_1.data
        w_total_loss_2 += w_loss_2.data

        s_loss_1, s_loss_2 = sent_loss(global_image_feature, sentence_feature, labels, class_ids, batch_size)
        s_total_loss_1 += s_loss_1.data
        s_total_loss_2 += s_loss_2.data

    w_cur_loss_1 = w_total_loss_1.item() / step
    w_cur_loss_2 = w_total_loss_2.item() / step
    s_cur_loss_1 = s_total_loss_1.item() / step
    s_cur_loss_2 = s_total_loss_2.item() / step

    print('| epoch {:4d}/{:4d} | eval valid loss | ''s_loss_1:{} | ''s_loss_2:{} | ''w_loss_1:{} | ''w_loss_2:{} |'
    .format(epoch, cfg.TRAIN.MAX_EPOCH, s_cur_loss_1, s_cur_loss_2, w_cur_loss_1, w_cur_loss_2))

    # attention Maps
    img_set, _ = build_super_images(real_imgs[-1].cpu(), captions, ixtoword, attn, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/eval_attention_maps_%04d.jpg' % (image_dir, epoch)
        im.save(fullpath)

    evaluate = [w_cur_loss_1, w_cur_loss_2, s_cur_loss_1, s_cur_loss_2]

    return evaluate

def test(dataloader, cnn_model, rnn_model, batch_size, epoch, ixtoword, image_dir):
    cnn_model.eval()
    rnn_model.eval()
    s_total_loss_1 = 0
    s_total_loss_2 = 0
    w_total_loss_1 = 0
    w_total_loss_2 = 0
    for step, data in enumerate(dataloader, 1):
        real_imgs, captions, cap_lens, class_ids, keys, captions_vector = prepare_data(data)

        local_image_features, global_image_feature = cnn_model(real_imgs[-1])
        # embedding_dim = local_image_features.size(1)
        embedding_dim, att_sze = local_image_features.size(1), local_image_features.size(2)
        # local_image_features = local_image_features.view(batch_size, embedding_dim, -1)

        hidden = rnn_model.init_hidden(batch_size)
        words_features, sentence_feature = rnn_model(captions, cap_lens, hidden, captions_vector)

        w_loss_1, w_loss_2, attn = words_loss(local_image_features, words_features, labels, cap_lens, class_ids, batch_size, epoch, mode='test')
        w_total_loss_1 += w_loss_1.data
        w_total_loss_2 += w_loss_2.data

        s_loss_1, s_loss_2 = sent_loss(global_image_feature, sentence_feature, labels, class_ids, batch_size)
        s_total_loss_1 += s_loss_1.data
        s_total_loss_2 += s_loss_2.data

    w_cur_loss_1 = w_total_loss_1.item() / step
    w_cur_loss_2 = w_total_loss_2.item() / step
    s_cur_loss_1 = s_total_loss_1.item() / step
    s_cur_loss_2 = s_total_loss_2.item() / step

    print('| epoch {:4d}/{:4d} | test valid loss | ''s_loss_1:{} | ''s_loss_2:{} | ''w_loss_1:{} | ''w_loss_2:{} |'
    .format(epoch, cfg.TRAIN.MAX_EPOCH, s_cur_loss_1, s_cur_loss_2, w_cur_loss_1, w_cur_loss_2))

    # attention Maps
    img_set, _ = build_super_images(real_imgs[-1].cpu(), captions, ixtoword, attn, att_sze)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/test_attention_maps_%04d.jpg' % (image_dir, epoch)
        im.save(fullpath)

    test = [w_cur_loss_1, w_cur_loss_2, s_cur_loss_1, s_cur_loss_2]

    return test

def build_models():
    # build model ############################################################
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    labels = Variable(torch.LongTensor(range(batch_size)))
    start_epoch = 1
    if cfg.TRAIN.NET_E != '':
        state_dict = torch.load(cfg.TRAIN.NET_E)
        text_encoder.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_E)
        #
        name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)
        start_epoch = int(cfg.TRAIN.NET_E[-8:-4])+1
        print('start_epoch', start_epoch)
    elif cfg.TRAIN.CONTINUE:
        output_dir = '../output/%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME)
        model_dir = os.path.join(output_dir, 'Model')
        model_file = glob.glob('%s/current/text_encoder_*.pth' % model_dir)[0]
        state_dict = torch.load(model_file)
        text_encoder.load_state_dict(state_dict)
        print('Load ', model_file)
        #
        name = model_file.replace('text_encoder', 'image_encoder')
        state_dict = torch.load(name)
        image_encoder.load_state_dict(state_dict)
        print('Load ', name)
        start_epoch = int(model_file[-8:-4]) + 1
        print('start_epoch', start_epoch)

    if cfg.CUDA:
        text_encoder = text_encoder.cuda()
        image_encoder = image_encoder.cuda()
        labels = labels.cuda()

    return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":

    start = time.time()

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    args.manualSeed = random.randint(1, 100)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    output_dir = '../output/%s_%s' % (cfg.DATASET_NAME, cfg.CONFIG_NAME)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    log_dir = os.path.join(output_dir, 'Log')
    numeric_dir = os.path.join(output_dir, 'Numeric')
    mkdir_p(model_dir)
    mkdir_p(image_dir)
    mkdir_p(log_dir)
    mkdir_p(numeric_dir)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    batch_size = cfg.TRAIN.BATCH_SIZE
    dataset = TextDataset(cfg.DATA_DIR, 'train', base_size=cfg.TREE.BASE_SIZE,
                          transform=transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize), transforms.RandomHorizontalFlip()]))
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #
    dataset_val = TextDataset(cfg.DATA_DIR, 'val', base_size=cfg.TREE.BASE_SIZE,
                              transform=transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize), transforms.RandomHorizontalFlip()]))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))

    # # test data #
    dataset_test = TextDataset(cfg.DATA_DIR, 'test', base_size=cfg.TREE.BASE_SIZE,
                              transform=transforms.Compose([transforms.Resize(imsize), transforms.CenterCrop(imsize)]))
    dataloader_test = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

    # Train ##############################################################
    text_encoder, image_encoder, labels, start_epoch = build_models()
    para = list(text_encoder.parameters())
    for v in image_encoder.parameters():
        if v.requires_grad:
            para.append(v)
    # optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        lr = cfg.TRAIN.ENCODER_LR
        logs = []
        timer = []
        for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH+1):
            epoch_start = time.time()
            optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
            print('-' * 150)
            training = train(dataloader, image_encoder, text_encoder, batch_size, labels, optimizer, epoch, dataset.ixtoword, image_dir)
            print('-' * 150)
            eval = evaluate(dataloader_val, image_encoder, text_encoder, batch_size, epoch, dataset.ixtoword, image_dir)
            print('-' * 150)
            test = test(dataloader_test, image_encoder, text_encoder, batch_size, epoch, dataset.ixtoword, image_dir)
            print('-' * 150)
            if lr > cfg.TRAIN.ENCODER_LR/10.:
                lr *= 0.98
            log = {'epoch': epoch,
            'train_w_loss_1': training[0], 'train_w_loss_2': training[1],
            'train_s_loss_1': training[2], 'train_s_loss_2': training[3],
            'eval_w_loss_1':eval[0], 'eval_w_loss_2':eval[1],
            'eval_s_loss_1':eval[2], 'eval_s_loss_2':eval[3],
            'test_w_loss_1':test[0], 'test_w_loss_2':test[1],
            'test_s_loss_1':test[2], 'test_s_loss_2':test[3]}
            logs.append(log)
            epoch_end = time.time()
            epoch_time = epoch_end-epoch_start
            tim =  {'epoch': epoch, 'epoch_time': epoch_time}
            timer.append(tim)
            print(f'epoch_time:{epoch_end-epoch_start}')

            if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or epoch == cfg.TRAIN.MAX_EPOCH):
                torch.save(image_encoder.state_dict(), '%s/image_encoder_%04d.pth' % (model_dir, epoch))
                torch.save(text_encoder.state_dict(), '%s/text_encoder_%04d.pth' % (model_dir, epoch))
                df = pd.DataFrame(logs)
                df.to_csv('%s/log_%04d.csv' %(log_dir, epoch), index=False)
                dt = pd.DataFrame(timer)
                dt.to_csv('%s/time_%04d.csv' %(log_dir, epoch), index=False)
                print('Save G/Ds models.')

        end = time.time()
        print(f'all_time:{end-start}')

    except KeyboardInterrupt:
        print('-' * 150)
        print('Exiting from training early')
