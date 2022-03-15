'''
some functions incl the ones from woz.py which turn the JSONs into txt files for us
'''

import os
import json
import re

def tag_sequence(sequence):
    '''

    method to take a list of tokens (sequence) and to replace years with <year> 1994,
    days with <days> monday, decimals with <decimal> 0.1 etc
    '''

    sequence = sequence.split(' ')
    # Now replace years, ints, decimals, days, numbers with tags
    sequence = [re.sub('^[12][0-9]{3}$', '<year>{}'.format(tok), tok) for tok in sequence]  # tag years
    sequence = [re.sub(r'\d{1,2}(?:(?:am|pm)|(?::\d{1,2})(?:am|pm)?)', '<time>{}'.format(tok), tok) for tok in sequence]  # tag times
    #sequence = [re.sub(r'^(\d{5}?[\s.-]?\d{6}%)', '<phone>{}'.format(tok), tok) for tok in sequence]  # tag phone numbers
    sequence = [re.sub('^[0-9]+', '<integer>{}'.format(tok), tok) for tok in sequence]  # tag integers
    sequence = [re.sub('^[0-9]+\.+[0-9]*$', '<decimal>{}'.format(tok), tok) for tok in sequence]  # tag decimals
    sequence = [re.sub('(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
                       '<dayofweek>{}'.format(tok), tok.lower()) for tok in sequence]  # tag days of week
    sequence = [re.sub('(january|february|march|april|may|june|july|august|september|october|november|december)',
                       '<month>{}'.format(tok), tok) for tok in sequence]  # tag month
    sequence = [re.sub('^[0-9]+(st|nd|rd|th)',
                       '<days>{}'.format(tok), tok) for tok in sequence]  # tag days (in date) - can have errors in this
    #sequence = [re.sub('^\s[0-9]', '<otherNum>{}'.format(tok), tok) for tok in sequence]  # tag all remaining numbers
    return ' '.join(sequence)


def make_woz_datasets(bKnowledge, tagging=False, situation='restaurant'):
    '''

    updated function to allow tagging. Tagging will tag all the occurences in each sequence with the tags outlined
    in the tag_sequence() function.
    '''

    if tagging:
        pre = 'wozTagged'
    else:
        pre = 'woz'

    if bKnowledge:
        out_names = [pre+'.train_c.txt', pre+'.valid_c.txt', pre+'.test_c.txt']
    else:
        out_names = [pre+'.train_a.txt', pre+'.valid_a.txt', pre+'.test_a.txt']

    max_ins = [18, 2, 2]
    subpaths = ['train', 'dev', 'test'] # filenames containing the data

    count = 0
    counts = []
    for dataset in range(len(out_names)):
        fout = open(out_names[dataset], 'wt')
        for dialog in range(1, max_ins[dataset], 1):
            file_name = os.path.join(subpaths[dataset], 'dialogues_%03d.json' % dialog)
            with open(file_name) as f:
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
                            if tagging:
                                utterance = tag_sequence(utterance)
                                print(utterance)
                                a = input('')

                            for frame in turn['frames']:
                                if frame['service'] == situation:
                                    knowledge = ''
                                    try:
                                        knowledge = '[KNOWLEDGE] '
                                        for slot in frame['slots']:
                                            temp = '%s [EQUAL] %s [SEP] ' % (slot['slot'], slot['value'])
                                            knowledge = knowledge + temp
                                    except:
                                        pass
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
                                            pass
                                    except:
                                        pass

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

make_woz_datasets(False, tagging=True)
