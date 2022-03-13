from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_metric
from pprint import pprint

import torch
import math
import time
import sys
import os
import json
import numpy as np


_datadir = './train/'

def make_woz_datasets(bKnowledge, situation='restaurant'):
    if bKnowledge:
        out_names = ['woz.train_c.txt', 'woz.valid_c.txt', 'woz.test_c.txt']
    else:
        out_names = ['woz.train_b.txt', 'woz.valid_a.txt', 'woz.test_a.txt', 'woz.valid_b.txt', 'woz.test_b.txt']
    max_ins = [18, 2, 2, 2, 2]

    count = 0
    counts = []
    for dataset in range(len(out_names)):
        fout = open(out_names[dataset], 'wt')
        for dialog in range(1, max_ins[dataset], 1):
            file_name = 'dialogues_%03d.json' % dialog
            path_to_file = os.path.join(_datadir, file_name)
            with open(path_to_file) as f:
                data = json.load(f)
            for dialogue in data:
                if len(dialogue['services']) == 1:
                    if dialogue['services'][0] == situation:
                        prev_speaker = ''
                        prev_utterance = ''
                        for turn in dialogue['turns']:
                            count = count + 1
                            speaker = turn['speaker']
                            utterance = turn['utterance']

                            for frame in turn['frames']:
                                if frame['service'] == situation:
                                    knowledge = ''
                                    try:
                                        knowledge = '[KNOWLEDGE] '
                                        for slot in frame['slots']:
                                            temp = '%s [EQUAL] %s [SEP] ' % (slot['slot'], slot['value'])
                                            knowledge = knowledge + temp
                                    except:
                                        nothing = 1
                                    try:
                                        if len(knowledge) == 0:
                                            knowledge = '[KNOWLEDGE] '
                                        try:
                                            intent = frame['state']['active_intent']
                                            temp = '%s [EQUAL] %s [SEP] ' % ('active_intent', intent)
                                            knowledge = knowledge + temp
                                            slot_values = frame['state']['slot_values']
                                            for slot in slot_values:
                                                vals = slot_values[slot]
                                                for val in vals:
                                                    temp = '%s [EQUAL] %s [SEP] ' % (slot, val)
                                                    knowledge = knowledge + temp
                                        except:
                                            nothing = 1
                                    except:
                                        noting = 1

                            if len(prev_speaker) > 0:
                                if not bKnowledge:
                                    knowledge = ''
                                if dataset == 0:
                                    text = '[%s] %s %s [%s] %s [END]' % (prev_speaker,
                                                                         prev_utterance,
                                                                         knowledge,
                                                                         speaker,
                                                                         utterance)
                                else:
                                    text = '[%s] %s %s [%s] | %s [END]' % (prev_speaker,
                                                                           prev_utterance,
                                                                           knowledge,
                                                                           speaker,
                                                                           utterance)
                                fout.write('%s\n' % (text))
                            prev_speaker = speaker
                            prev_utterance = utterance
        counts.append(count)
        count = 0
    print(counts)



def main(situation='restaurant', model_path='gpt2', test_name='woz.test_a.txt', gen_mode=0):
    make_woz_datasets(True, situation=situation)
    make_woz_datasets(False, situation=situation)

    gen_labels = ['logits', 'greedy', 'beam', 'top-p']

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

    if torch.cuda.is_available():
        model = model.cuda()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total parameters = ', params)

    metrics = {
               'bleu':load_metric('bleu'),
               'meteor':load_metric('meteor'),
                'google_bleu':load_metric('google_bleu')
    }

    predicts = []
    refs = []
    best = []
    metric_results = dict.fromkeys(metrics.keys(), [])
    max_len = 0
    total = 0
    with open(test_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            text = line.split('|')
            prompt = text[0].strip(' ')
            in_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if len(in_ids) > max_len:
                max_len = len(in_ids)
            total = total + 1

    max_len = max_len + 32
    print('max_len: %d total: %d' % (max_len, total))

    obs = 0
    with open(test_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            text = line.split('|')
            prompt = text[0].strip(' ')
            ref = text[1].strip(' ')
            obs = obs + 1
            in_ids = tokenizer.encode(prompt, add_special_tokens=True)

            if gen_mode == 0:
                seq_len = 0
                bDone = False
                while not bDone:
                    input_ids = torch.tensor(in_ids).unsqueeze(0)
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                    outputs = model(input_ids, labels=input_ids)
                    decoded = []
                    for i in range(outputs[1].size(1)):
                        decoded.append(torch.argmax(outputs[1][0][i][:]).item())
                    decoded = torch.tensor(decoded)
                    if torch.cuda.is_available():
                        decoded = decoded.cuda()
                    in_ids.append(decoded[decoded.size(0) - 1].item())
                    input_ids = torch.tensor(in_ids).unsqueeze(0)
                    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    tokens = text.split(' ')
                    if tokens[len(tokens) - 1] == '[END]':
                        bDone = True
                    if len(tokens) >= max_len:
                        bDone = True

            if gen_mode == 1:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                greedy = model.generate(input_ids, max_length=max_len)
                text2 = tokenizer.decode(greedy[0], skip_special_tokens=False)
                tokens = text2.split()

            if gen_mode == 2:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                beam = model.generate(input_ids, max_length=max_len, num_beams=5, early_stopping=True)
                text2 = tokenizer.decode(beam[0], skip_special_tokens=False)
                tokens = text2.split()

            if gen_mode == 3:
                input_ids = torch.tensor(in_ids).unsqueeze(0)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                top_p = model.generate(input_ids, max_length=max_len, do_sample=True, top_p=0.90, top_k=0)
                text2 = tokenizer.decode(top_p[0], skip_special_tokens=False)
                tokens = text2.split()

            first = len(prompt.split(' '))
            try:
                pos_end = tokens.index('[END]')
            except:
                pos_end = len(tokens)
            try:
                pos_enduser = tokens.index('[END][USER]')
            except:
                pos_enduser = len(tokens)
            try:
                pos_endsystem = tokens.index('[END][SYSTEM]')
            except:
                pos_endsystem = len(tokens)
            last = min(pos_end, pos_enduser, pos_endsystem, len(tokens))
            predict = ' '.join(tokens[first:last])

            predictions = [predict.split()]
            references = [[ref.split()]]
            predicts.append(predictions)
            refs.append(references)

            for metric in metrics:
                M = metrics[metric]
                try:
                    results = M.compute(predictions=predictions, references=references)
                    metric_results[metric] = metric_results[metric] + [results[metric]]
                except:
                    pass

            #if obs > 3:
            #    break # just to speed it up if needed

    print('\n------')
    print("Situation: {}".format(situation))

    for metric in metrics:
        results = metrics[metric].compute(predictions=predicts, references=refs)
        print('Final %s on %s %s: %7.3f % 7.3f' % (gen_labels[gen_mode], metric, test_name, results[metric], sum(metric_results[metric]) / 511.0))
    print(len(predicts), len(refs))
    print(' ')


if __name__ == "__main__":

    main(situation='restaurant')
    main(situation='hotel')
    #main(situation='train')

    #make_tagged_datasets()
