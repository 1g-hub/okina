from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import gensim.downloader

import os
import numpy as np
import pandas as pd
from PIL import Image

import numpy.random as random
import pickle

def prepare_data(data):
    imgs, captions, captions_lens, class_ids, keys, captions_vector = data
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_lens, 0, True)
    real_imgs = []
    for i in range(len(imgs)):
        imgs[i] = imgs[i][sorted_cap_indices]
        if cfg.CUDA:
            real_imgs.append(Variable(imgs[i]).cuda())
        else:
            real_imgs.append(Variable(imgs[i]))

    captions = captions[sorted_cap_indices].squeeze()
    class_ids = class_ids[sorted_cap_indices].numpy()
    captions_vector = captions_vector[sorted_cap_indices]
    # sent_indices = sent_indices[sorted_cap_indices]
    keys = [keys[i] for i in sorted_cap_indices.numpy()]
    # print('keys', type(keys), keys[-1])  # list
    if cfg.CUDA:
        captions = Variable(captions).cuda()
        sorted_cap_lens = Variable(sorted_cap_lens).cuda()
        captions_vector = Variable(captions_vector).cuda()
    else:
        captions = Variable(captions)
        sorted_cap_lens = Variable(sorted_cap_lens)
        captions_vector = Variable(captions_vector)

    return [real_imgs, captions, sorted_cap_lens, class_ids, keys, captions_vector]


def get_imgs(img_path, imsize, bbox=None, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])

    if transform is not None:
        img = transform(img)

    ret = []
    for i in range(cfg.TREE.BRANCH_NUM):
        # print(imsize[i])
        if i < (cfg.TREE.BRANCH_NUM - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        ret.append(normalize(re_img))

    return ret

class TextDataset(data.Dataset):
    def __init__(self, data_dir, split='train', base_size=64, transform=None, target_transform=None):
        self.transform = transform
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform
        self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

        self.imsize = []
        for i in range(cfg.TREE.BRANCH_NUM):
            self.imsize.append(base_size)
            base_size = base_size * 2

        self.data = []
        self.data_dir = data_dir
        if data_dir.find('concept') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        split_dir = os.path.join(data_dir, split)

        self.filenames, self.captions, self.ixtoword, self.wordtoix, self.ixtovector, self.n_words = self.load_text_data(data_dir, split)

        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,delim_whitespace=True, header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'images.txt')
        df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        print('Total file_num: ', len(filenames))
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().split('\n')
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
        return all_captions

    def build_dictionary(self, train_captions, val_captions, test_captions):
        pretrained_model = cfg.TEXT.PRETRAINED_MODEL
        model = gensim.downloader.load(pretrained_model)
        langs = ['en', 'fr', 'de', 'ja', 'it', 'fi', 'zh', 'pt', 'la', 'nl', 'es', 'ru', 'sh', 'sv', 'cs', 'pl', 'ms',
                 'bg', 'ca', 'eo', 'no', 'is', 'sl', 'ro', 'ga', 'vi', 'lv', 'grc', 'tr', 'da', 'ar', 'fa', 'ko', 'hy',
                 'eu', 'fro', 'io', 'oc', 'gd', 'gl', 'nrf', 'th', 'ka', 'he', 'sq', 'fo', 'te', 'mk', 'se', 'mul',
                 'et', 'gv', 'sk', 'xcl', 'hi', 'af', 'ang', 'lt', 'ast', 'uk', 'cy', 'nv', 'mg', 'kk', 'rup', 'sa',
                 'non', 'vo', 've', 'sw', 'ur', 'ku', 'fil', 'az', 'ta', 'hsb']
        print('Load pretrained model from ', pretrained_model)
        word_counts = defaultdict(float)
        captions = train_captions + val_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[-1] = '<end>'
        ixtoword[-2] = '<unknown>'
        ixtoword[-3] = '<number>'


        ixtovector = {}
        ixtovector[-1] = torch.zeros(cfg.TEXT.EMBEDDING_DIM,)
        if pretrained_model == 'conceptnet-numberbatch-17-06-300':
            ixtovector[-2] = torch.from_numpy(model.wv['/c/en/unknown'])
            ixtovector[-3] = torch.from_numpy(model.wv['/c/en/number'])
        else:
            ixtovector[-2] = torch.from_numpy(model.wv['unknown'])
            ixtovector[-3] = torch.from_numpy(model.wv['number'])

        wordtoix = {}
        wordtoix['<end>'] = -1
        wordtoix['<unknown>'] = -2
        wordtoix['<number>'] = -3

        for w in vocab:
            try:
                for lang in langs:
                    if pretrained_model == 'conceptnet-numberbatch-17-06-300':
                        s = '/c/%s/%s' %(lang, w)
                        wordtoix[w] = model.vocab[s].index
                        ixtovector[wordtoix[w]] = torch.from_numpy(model.wv[s])
                        ixtoword[wordtoix[w]] = w
                        break
                    else:
                        wordtoix[w] = model.vocab[w].index
                        ixtovector[model.vocab[w].index] = torch.from_numpy(model.wv[w])
                        ixtoword[model.vocab[w].index] = w
                        break
            except:
                if w.isnumeric():
                    wordtoix[w] = -3
                else:
                    wordtoix[w] = -2

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            train_captions_new.append(rev)

        val_captions_new = []
        for t in val_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            val_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            test_captions_new.append(rev)

        return [train_captions_new, val_captions_new, test_captions_new, ixtoword, wordtoix, ixtovector, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_fasttext.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        val_names = self.load_filenames(data_dir, 'val')
        test_names = self.load_filenames(data_dir, 'test')

        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, val_captions, test_captions = x[0], x[1], x[2]
                ixtoword, wordtoix, ixtovector = x[3], x[4],m x[5]
                del x
                n_words = len(ixtoword)
                print('Load from: ', filepath)

        else:
            train_captions = self.load_captions(data_dir, train_names)
            val_captions = self.load_captions(data_dir, val_names)
            test_captions = self.load_captions(data_dir, test_names)
            train_captions, val_captions, test_captions, ixtoword, wordtoix, ixtovector, n_words = self.build_dictionary(train_captions, val_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, val_captions, test_captions, ixtoword, wordtoix, ixtovector], f, protocol=2)
                print('Save to: ', filepath)

        if split == 'train':
            captions = train_captions
            filenames = train_names
        elif split == 'val':
                captions = val_captions
                filenames = val_names
        elif split == 'test':
            captions = test_captions
            filenames = test_names

        return filenames, captions, ixtoword, wordtoix, ixtovector, n_words

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        x = np.ones((cfg.TEXT.WORDS_NUM, 1), dtype='int64') * (-1)
        x_len = num_words
        if num_words <= cfg.TEXT.WORDS_NUM:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:cfg.TEXT.WORDS_NUM]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = cfg.TEXT.WORDS_NUM
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = index
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
            data_dir = '%s' % self.data_dir
        else:
            bbox = None
            data_dir = self.data_dir
        #
        img_name = '%s/images/%s.jpg' % (data_dir, key)
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm)
        # random select a sentence
        if self.embeddings_num > 1:
            sent_ix = random.randint(0, self.embeddings_num)
            new_sent_ix = index * self.embeddings_num + sent_ix
        elif self.embeddings_num == 1:
            new_sent_ix = index
        caps, cap_len = self.get_caption(new_sent_ix)

        captions_vector = torch.zeros(cfg.TEXT.WORDS_NUM, cfg.TEXT.EMBEDDING_DIM)
        for i in range(cfg.TEXT.WORDS_NUM):
            captions_vector[i] = self.ixtovector[caps[i].item()]

        return imgs, caps, cap_len, cls_id, key, captions_vector

    def __len__(self):
        return len(self.filenames)
