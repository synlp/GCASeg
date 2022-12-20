from cws_model import CWS
from modules import BertTokenizer

import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.nn import CrossEntropyLoss

from data_utils import DataProcessor
from cws_helper import get_label_list
from cws_eval import cws_evaluate_word_PRF

from modules.optimization import BertAdam
from modules.schedulers import LinearWarmUpScheduler
# from apex import amp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

class NodeInstructor():
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        self.label_list = ["B", "I", "E", "S", "[CLS]", "[SEP]"]
        self.label_map = {label: i for i, label in enumerate(self.label_list, 1)}
        self.num_labels = len(self.label_map)+1
        gram2id, embedding = CWS.read_pretrained_embedding(args.word_embeddings)
        self.data_processor = DataProcessor(self.tokenizer, self.label_map, args.use_gca,
                                            gram2id=gram2id, max_ngram_length=args.max_ngram_length)
        self.train_features, self.dev_features = self.data_processor.get_features(args.data_dir, dataset)
        # self.num_labels = self.data_processor.get_tag_size()

    def fetch_train_feature(self):
        while True:
            sample_index = list(range(len(self.train_features)))
            random.shuffle(sample_index)
            for index in sample_index:
                feature = self.train_features[index].copy()
                del feature["label_id"]
                feature["index"] = index
                yield feature

    def transmit_encrypted_data_for_train(self, batch_size=1):
        return [next(self.fetch_train_feature()) for i in range(batch_size)]

    def transmit_inf_results_and_ret_loss(self, results):
        attention_mask = torch.stack([self.train_features[res["index"]]["label_mask"] for res in results], dim=0).to(self.device)
        labels = torch.stack([self.train_features[res["index"]]["label_id"] for res in results], dim=0).to(self.device)
        logits = torch.stack([res["logits"] for res in results], dim=0)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss_fct = CrossEntropyLoss()
        return loss_fct(active_logits, active_labels)

    def transmit_encrypted_data_for_eval(self):
        features = []
        for index,feature in enumerate(self.dev_features):
            feature = feature.copy()
            del feature["label_id"]
            feature["index"] = index
            features.append(feature)
        return features

    def transmit_inf_results_for_eval(self, t_outputs_all):
        t_targets_all = [feature["label_id"].tolist() for feature in self.dev_features]

        precision, recall, f1 = cws_evaluate_word_PRF(t_outputs_all, t_targets_all, self.label_list)

        return {
            "precision": precision,
            "f1": f1,
            "recall":recall
        }

class Instructor:
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    def saving_model(self, saving_model_path, model, optimizer):
        if not os.path.exists(saving_model_path):
            os.mkdir(saving_model_path)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(saving_model_path, CONFIG_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, "w", encoding='utf-8') as writer:
            writer.write(model_to_save.config.to_json_string())
        torch.save({'optimizer': optimizer.state_dict(),
                    'master params': list(optimizer)},
                   output_optimizer_file)

    def load_model(self, model, optimizer, saving_model_path):
        output_model_file = os.path.join(saving_model_path, WEIGHTS_NAME)
        output_optimizer_file = os.path.join(saving_model_path, "optimizer.pt")
        #model
        checkpoint_model = torch.load(output_model_file, map_location="cpu")
        model.load_state_dict(checkpoint_model)
        #optimizer
        checkpoint_optimizer = torch.load(output_optimizer_file, map_location="cpu")
        if self.args.fp16:
            from apex import amp
            optimizer._lazy_init_maybe_master_weights()
            optimizer._amp_stash.lazy_init_called = True
            optimizer.load_state_dict(checkpoint_optimizer['optimizer'])
            for param, saved_param in zip(amp.master_params(optimizer), checkpoint_optimizer['master params']):
                param.data.copy_(saved_param.data)
        else:
            optimizer.load_state_dict(checkpoint_optimizer["optimizer"])
        return model, optimizer

    def save_args(self):
        output_args_file = os.path.join(self.args.outdir, 'training_args.bin')
        torch.save(self.args, output_args_file)

    def model_inference(self, model, features):
        input_ids = torch.stack([feature["input_ids"] for feature in features], dim=0).to(self.args.device)
        input_mask = torch.stack([feature["input_mask"] for feature in features], dim=0).to(self.args.device)
        segment_ids = torch.stack([feature["segment_ids"] for feature in features], dim=0).to(self.args.device)
        valid_ids = torch.stack([feature["valid_ids"] for feature in features], dim=0).to(self.args.device)
        label_mask = torch.stack([feature["label_mask"] for feature in features], dim=0).to(self.args.device)

        if self.args.use_gca is not None:
            word_ids = torch.stack([feature["word_ids"] for feature in features], dim=0).to(self.args.device)
            matching_matrix = torch.stack([feature["matching_matrix"] for feature in features], dim=0).to(self.args.device)
        else:
            word_ids = None
            matching_matrix = None

        tag_seq, logits = model(input_ids, segment_ids, input_mask, valid_ids, label_mask,
            word_seq=word_ids, label_value_matrix=matching_matrix)
        for i in range(len(features)):
            features[i]["logits"] = logits[i]
            features[i]["tag_seq"] = tag_seq[i].cpu().tolist()
        return features

    def run(self):
        self.save_args()
        args = self.args
        self.label_list = ["B", "I", "E", "S", "[CLS]", "[SEP]"]
        self.label_map = {label: i for i, label in enumerate(self.label_list, 1)}
        model = CWS(self.args, self.label_map, args.bert_model, word_embedding=args.word_embeddings)
        model = model.to(args.device)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=args.total_step)

        node_dict = {}
        dataset_list = args.dataset.split(",")
        print("dataset: {}".format(args.dataset))
        for dataset in dataset_list:
            node_dict[dataset] = NodeInstructor(args, dataset, args.device)

        for _ in trange(args.total_step, desc="Step"):
            model.train()
            random.shuffle(dataset_list)
            cand_dataset_list = dataset_list[:random.randint(1, len(dataset_list)+1)]
            loss = 0
            for dataset in cand_dataset_list:
                train_features = node_dict[dataset].transmit_encrypted_data_for_train(batch_size=random.randint(1,args.batch_size+1))
                features = self.model_inference(model, train_features)
                _loss = node_dict[dataset].transmit_inf_results_and_ret_loss(features)
                loss += _loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            for dataset in dataset_list:
                eval_features = node_dict[dataset].transmit_encrypted_data_for_eval()
                features = self.model_inference(model, eval_features)
                t_outputs_all = [feature["tag_seq"] for feature in features]
                result = node_dict[dataset].transmit_inf_results_for_eval(t_outputs_all)
                print(dataset)
                print(result)

def get_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--embedding', default='embedding', type=str)
    parser.add_argument('--word_embeddings', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--encoder', default='bilstm', type=str)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default='2e-5', type=float)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--bert_dropout', default=0.2, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_epoch', default=2, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--log', default='log', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--bert_dim', default=1024, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--max_ngram_length', default=5, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--bert_model', default='./bert-large-uncased', type=str)
    parser.add_argument('--outdir', default='./', type=str)
    parser.add_argument('--tool', default='stanford', type=str)
    parser.add_argument('--warmup_proportion', default=0.06, type=float)
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--loss_scale', default=0, type=int)
    parser.add_argument('--save', action='store_true', help="Whether to save model")
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument("--rank", type=int, default=0, help="local_rank for distributed training on gpus")
    parser.add_argument("--world_size", type=int, default=1, help="local_rank for distributed training on gpus")
    parser.add_argument("--init_method", type=str, default="", help="init_method")
    parser.add_argument("--total_step", type=int, default=10, help="total step")
    parser.add_argument("--sample_step", type=int, default=1, help="sample step")
    parser.add_argument("--use_bert",
                        action='store_true',
                        help="Whether to use BERT.")
    parser.add_argument("--use_zen",
                        action='store_true',
                        help="Whether to use ZEN.")
    parser.add_argument("--use_crf",
                        action='store_true',
                        help="Whether to use crf.")
    parser.add_argument("--use_gca",
                        action='store_true',
                        help="Whether to use global character associations.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to do_lower_case.")
    args = parser.parse_args()

    args.initializer = torch.nn.init.xavier_uniform_

    return args

def main():
    args = get_args()

    import datetime
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))
    args.outdir = os.path.join(args.outdir, "{}_bts_{}_lr_{}_warmup_{}_seed_{}_bert_dropout_{}_{}".format(
        args.dataset,
        args.batch_size,
        args.learning_rate,
        args.warmup_proportion,
        args.seed,
        args.bert_dropout,
        now_time
    ))
    if not os.path.exists(args.outdir):
        try:
            os.mkdir(args.outdir)
        except Exception as e:
            print(str(e))

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    log_file = '{}/{}-{}.log'.format(args.log, args.dataset, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()
