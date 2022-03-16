from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
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

        print('hi')
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx): # format seems to be required by GPT2, see https://discuss.huggingface.co/t/finetuning-gpt2-with-user-defined-loss/163/20?page=2
        dict = {"input_ids": self.x[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.y[idx]}
        return dict


def main(train_name='woz.train_a.txt', valid_name='woz.valid_a.txt'):
    '''
    Just a wrapper so I can run this in collab
    '''
    base_model = 'gpt2'
    torch.manual_seed(1)
    tokenizer = GPT2Tokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token # set the padding token
    model = GPT2LMHeadModel.from_pretrained(base_model, pad_token_id=tokenizer.eos_token_id)
    model.resize_token_embeddings(len(tokenizer))
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
            if True: #line.split(' ')[0] == '[USER]': # only train on replies by [SYSTEM] (that start with prompt by [USER])
                line = line.replace('\n', '')
                end_token = "[SYSTEM]" if line.split(' ')[0] == '[USER]' else "[USER]"
                SYS_index = line.index(end_token)
                prompt = line[:SYS_index + len(end_token)].strip(' ')
                resp = line[SYS_index + len(end_token):].strip(' ')
                resp = resp.split('  <|endoftext|>')[0]
                train_x.append(prompt)
                full = prompt + resp
                if len(full)>90: # trim bc sometimes they can be too big
                  full = full[:90]
                train_y.append(full)

    dataset_t = WozDataset(train_x, train_y, tokenizer=tokenizer)

    with open(valid_name, 'rt') as f:
        for line in f:
            if True: # line.split(' ')[0] == '[USER]': # only train on replies by [SYSTEM] (that start with prompt by [USER])
                line = line.replace('\n', '')
                end_token = "[SYSTEM]" if line.split(' ')[0] == '[USER]' else "[USER]"
                SYS_index = line.index(end_token)
                prompt = line[:SYS_index].strip(' ')
                resp = line[SYS_index:].strip(' ')
                resp = resp.split('  <|endoftext|>')[0]
                val_x.append(prompt)
                full = prompt + resp
                if len(full)>90:
                  full = full[:90]
                val_y.append(full)

    dataset_v = WozDataset(val_x, val_y, tokenizer=tokenizer)

    # config for training
    config = TrainingArguments(output_dir='./results/', num_train_epochs=5, logging_steps=20,
                               load_best_model_at_end=False, save_strategy="epoch",
                               per_device_train_batch_size=8, evaluation_strategy="steps",
                               warmup_steps=100, weight_decay=0.01, logging_dir='./Logs',
                               learning_rate=5e-6, do_train=True, do_eval=True)

    # start training
    #Trainer(model=model, args=config, train_dataset=dataset_t).train()
    Trainer(model=model, args=config, train_dataset=dataset_t, eval_dataset=dataset_v).train()

if __name__ == "__main__":
    main()

