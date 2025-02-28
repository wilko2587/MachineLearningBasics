from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# implemented with inspo from https://towardsdatascience.com/guide-to-fine-tuning-text-generation-models-gpt-2-gpt-neo-and-t5-dc5de6b3bc5e
# API key: 71d15fe01d1f78ec26109a22bfb225eae26a777e

class WozDataset(Dataset): # basic torch dataset to let the trainer function
    def __init__(self, prompt, resp, tokenizer): # prompt and response text as inputs

        # initializations
        self.x = []
        self.y = []
        self.attention_mask = []

        assert len(prompt) == len(resp)

        for i in range(len(prompt)):
            in_encoded_dict = tokenizer(prompt[i], add_special_tokens=True, padding="max_length",
                                        max_length=96)  # pad to make sure all inputs are same dimension
            out_encoded_dict = tokenizer(resp[i], add_special_tokens=True, padding="max_length", max_length=96)

            self.x.append(torch.tensor(in_encoded_dict['input_ids']))
            self.attention_mask.append(torch.tensor(in_encoded_dict['attention_mask']))
            self.y.append(torch.tensor(out_encoded_dict['input_ids']))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): # format seems to be required by GPT2, see https://discuss.huggingface.co/t/finetuning-gpt2-with-user-defined-loss/163/20?page=2
        dict = {"input_ids": self.x[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.y[idx]}
        return dict

def main():
    '''
    Just a wrapper so I can run this in collab
    '''
    base_model = 't5-base'
    torch.manual_seed(1)
    train_name = 'woz.train_b.txt'
    valid_name = 'woz.valid_b.txt'
    tokenizer = T5Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token  # set the padding token
    model = T5ForConditionalGeneration.from_pretrained(base_model, pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        model = model.cuda()

    # format and tokenize the training dataset
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    #create train and validation datasets
    with open(train_name, 'rt') as f:
        for line in f:
            if line.split(' ')[0] == '[USER]': # only train on replies by [SYSTEM] (that start with prompt by [USER])
                line = line.replace('\n', '')
                SYS_index = line.index('[SYSTEM]')
                prompt = line[:SYS_index].strip(' ')
                resp = line[SYS_index:].strip(' ')
                train_x.append(prompt)
                train_y.append(resp)

    dataset_t = WozDataset(train_x, train_y, tokenizer=tokenizer)

    with open(valid_name, 'rt') as f:
        for line in f:
            if line.split(' ')[0] == '[USER]': # only train on replies by [SYSTEM] (that start with prompt by [USER])
                line = line.replace('\n', '')
                SYS_index = line.index('[SYSTEM]')
                prompt = line[:SYS_index].strip(' ')
                resp = line[SYS_index:].strip(' ')
                val_x.append(prompt)
                val_y.append(resp)

    dataset_v = WozDataset(val_x, val_y, tokenizer=tokenizer)

    # config for training
    config = TrainingArguments(output_dir='./results/', num_train_epochs=10, logging_steps=50,
                               load_best_model_at_end=False, save_strategy="epoch",
                               per_device_train_batch_size=8,
                               warmup_steps=100, learning_rate = 7e-4, weight_decay=0.05, logging_dir='./Logs')

    # start training
    #Trainer(model=model, args=config, train_dataset=dataset_t).train()
    Trainer(model=model, args=config, train_dataset=dataset_t).train()
    model.save_pretrained('./models/') # ??? I think this is how you save a model??

