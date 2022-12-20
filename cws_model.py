from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch import nn
from modules import BertModel, ZenModel, BertTokenizer, CRF
from util import ZenNgramDict
from cws_helper import save_json, load_json
import subprocess
import os


DEFAULT_HPARA = {
    'max_seq_length': 500,
    'max_ngram_length': 5,
    'use_bert': False,
    'use_zen': False,
    'do_lower_case': False,
    'use_crf': False,
    'use_gca': False,
    'embedding_dim': 200,
}


class GCA(nn.Module):
    def __init__(self, hidden_size, word_size=None, embedding_dim=None, embedding=None):
        super(GCA, self).__init__()
        self.temper = hidden_size ** 0.5

        if embedding is not None:
            embedding = torch.tensor(embedding)
            embedding_dim = embedding.shape[1]
            self.word_embedding_a = nn.Embedding.from_pretrained(embedding, freeze=True)
        else:
            self.word_embedding_a = nn.Embedding(word_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, hidden_size, bias=True)
        self.word_embedding_c = nn.Embedding(6, hidden_size)

    def forward(self, word_seq, hidden_state, label_value_matrix):
        embedding_a = self.word_embedding_a(word_seq)
        embedding_c = self.word_embedding_c(label_value_matrix)

        embedding_a = self.linear(embedding_a)
        embedding_a = embedding_a.permute(0, 2, 1)
        u = torch.matmul(hidden_state, embedding_a) / self.temper

        label_value_matrix = label_value_matrix.clone().detach().to(label_value_matrix.device)

        tmp_word_mask_metrix = torch.clamp(label_value_matrix, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        sum_delta_exp_u = torch.stack([torch.sum(delta_exp_u, 2)] * delta_exp_u.shape[2], 2)

        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        embedding_c = embedding_c.permute(3, 0, 1, 2)
        o = torch.mul(p, embedding_c)

        o = o.permute(1, 2, 3, 0)
        o = torch.sum(o, 2)

        return o


class CWS(nn.Module):

    def __init__(self, args, labelmap, model_path, word_embedding=None, gram2id=None):
        super().__init__()
        self.labelmap = labelmap
        self.hpara = self.init_hyper_parameters(args)
        self.num_labels = len(self.labelmap) + 1
        self.max_seq_length = self.hpara['max_seq_length']
        self.max_ngram_length = self.hpara['max_ngram_length']
        self.use_crf = self.hpara['use_crf']
        self.use_gca = self.hpara['use_gca']

        from_pretrained = True

        if word_embedding is not None and gram2id is not None:
            raise ValueError()

        if self.hpara['use_zen']:
            raise ValueError()

        self.tokenizer = None
        self.bert = None
        self.zen = None
        self.zen_ngram_dict = None

        if self.hpara['use_bert']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            if from_pretrained:
                self.bert = BertModel.from_pretrained(model_path, cache_dir='')
            else:
                from modules import CONFIG_NAME, BertConfig
                config_file = os.path.join(model_path, CONFIG_NAME)
                config = BertConfig.from_json_file(config_file)
                self.bert = BertModel(config)
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        elif self.hpara['use_zen']:
            self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=self.hpara['do_lower_case'])
            self.zen_ngram_dict = ZenNgramDict(model_path, tokenizer=self.zen_tokenizer)
            self.zen = ZenModel.from_pretrained(model_path, cache_dir='')
            hidden_size = self.zen.config.hidden_size
            self.dropout = nn.Dropout(self.zen.config.hidden_dropout_prob)
        else:
            raise ValueError()

        if self.use_gca:
            if word_embedding is not None:
                self.gram2id, embedding = self.read_pretrained_embedding(word_embedding)
                self.hpara['embedding_dim'] = len(embedding[0])
                self.gca = GCA(hidden_size, embedding=embedding)
            elif gram2id is not None:
                self.gram2id = gram2id
                self.gca = GCA(hidden_size,
                               word_size=len(self.gram2id), embedding_dim=self.hpara['embedding_dim'])
        else:
            self.gram2id = None
            self.gca = None

        self.classifier = nn.Linear(hidden_size, self.num_labels, bias=True)

        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
        else:
            self.crf = None
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, valid_ids=None,
                attention_mask_label=None, labels=None,
                word_seq=None, label_value_matrix=None,
                input_ngram_ids=None, ngram_position_matrix=None,
                ):

        if self.bert is not None:
            sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        elif self.zen is not None:
            sequence_output, _ = self.zen(input_ids, input_ngram_ids=input_ngram_ids,
                                          ngram_position_matrix=ngram_position_matrix,
                                          token_type_ids=token_type_ids, attention_mask=attention_mask,
                                          output_all_encoded_layers=False)
        else:
            raise ValueError()

        batch_size, _, feat_dim = sequence_output.shape
        valid_output = self.dropout(sequence_output)

        if self.gca is not None:
            o = self.gca(word_seq, valid_output, label_value_matrix)
            o = self.dropout(o)

            valid_output = torch.add(o, valid_output)

        pred = self.classifier(valid_output)

        if labels is not None:
            if self.crf is not None:
                return -1 * self.crf(emissions=pred, tags=labels, mask=attention_mask_label)
            else:
                p_labels = pred[attention_mask_label]
                labels = labels[attention_mask_label]
                return self.loss_function(p_labels, labels)
        else:
            if self.crf is not None:
                return self.crf.decode(pred, attention_mask_label)[0]
            else:
                pre_labels = torch.argmax(pred, dim=2)
                return pre_labels, pred


    @staticmethod
    def init_hyper_parameters(args):
        hyper_parameters = DEFAULT_HPARA.copy()
        hyper_parameters['max_seq_length'] = args.max_seq_len
        hyper_parameters['max_ngram_length'] = args.max_ngram_length

        hyper_parameters['use_bert'] = args.use_bert
        hyper_parameters['use_zen'] = args.use_zen
        hyper_parameters['do_lower_case'] = args.do_lower_case

        hyper_parameters['use_crf'] = args.use_crf
        hyper_parameters['use_gca'] = args.use_gca

        return hyper_parameters

    @property
    def model(self):
        return self.state_dict()

    def save_model(self, output_model_dir, vocab_dir):
        best_eval_model_dir = os.path.join(output_model_dir, 'model')
        if not os.path.exists(best_eval_model_dir):
            os.makedirs(best_eval_model_dir)

        output_model_path = os.path.join(best_eval_model_dir, 'pytorch_model.bin')
        torch.save(self.state_dict(), output_model_path)

        output_tag_file = os.path.join(best_eval_model_dir, 'labelset.json')
        save_json(output_tag_file, self.labelmap)

        output_hpara_file = os.path.join(best_eval_model_dir, 'hpara.json')
        save_json(output_hpara_file, self.hpara)

        if self.gram2id is not None:
            output_gram_file = os.path.join(best_eval_model_dir, 'gram2id.json')
            save_json(output_gram_file, self.gram2id)

        output_config_file = os.path.join(best_eval_model_dir, 'config.json')
        with open(output_config_file, "w", encoding='utf-8') as writer:
            if self.bert:
                writer.write(self.bert.config.to_json_string())
            elif self.zen:
                writer.write(self.zen.config.to_json_string())
        output_bert_config_file = os.path.join(best_eval_model_dir, 'bert_config.json')
        command = 'cp ' + str(output_config_file) + ' ' + str(output_bert_config_file)
        subprocess.run(command, shell=True)

        if self.bert:
            vocab_name = 'vocab.txt'
        elif self.zen:
            vocab_name = 'vocab.txt'
        else:
            raise ValueError()
        vocab_path = os.path.join(vocab_dir, vocab_name)
        command = 'cp ' + str(vocab_path) + ' ' + str(os.path.join(best_eval_model_dir, vocab_name))
        subprocess.run(command, shell=True)

    @classmethod
    def load_model(cls, model_path, device):
        tag_file = os.path.join(model_path, 'labelset.json')
        labelmap = load_json(tag_file)

        hpara_file = os.path.join(model_path, 'hpara.json')
        hpara = load_json(hpara_file)
        DEFAULT_HPARA.update(hpara)

        gram_file = os.path.join(model_path, 'gram2id.json')
        if os.path.exists(gram_file):
            gram2id = load_json(gram_file)
        else:
            gram2id = None

        res = cls(labelmap=labelmap, hpara=DEFAULT_HPARA, model_path=model_path, gram2id=gram2id, word_embedding=None)
        res.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device))
        return res


    @staticmethod
    def read_pretrained_embedding(embedding_file, pad=True):
        word_embedding = []
        word2id = {}
        index = 0

        with open(embedding_file, 'r', encoding='utf8') as f:
            lines = f.readlines()

        embedding_dim = len(lines[0].strip().split()) - 1

        if pad:
            word2id['<PAD>'] = index
            index += 1
            word_embedding.append(np.random.randn(embedding_dim).tolist())

        for line in lines:
            line = line.strip()
            if line == '':
                continue
            splits = line.split()

            if not len(splits) == embedding_dim + 1:
                print(line)
                raise ValueError()

            word2id[splits[0]] = index
            index += 1

            emb_vec = [float(v) for v in splits[1:]]

            word_embedding.append(emb_vec)

        return word2id, word_embedding



