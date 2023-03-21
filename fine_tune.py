import transformers 
import torch 
import torch.nn as nn 
import peft 
from peft import LoraConfig
from chatglm import ChatGLMTokenizer, ChatGLMForConditionalGeneration 
from dataset import TextDataSet
import os 
import loralib as lora
import numpy as np 
import tqdm 



ckpt_path = './ckpt'
tokenizer = ChatGLMTokenizer.from_pretrained(ckpt_path)
model = ChatGLMForConditionalGeneration.from_pretrained(ckpt_path)


config = LoraConfig(
            peft_type="LORA", 
            task_type="SEQ_2_SEQ_LM", 
            r=32, 
            lora_alpha=32, 
            target_modules=["q", "k", "v"],
            lora_dropout=0.1, 
            )



class QKV_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(QKV_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features//3)
        self.linear_k = torch.nn.Linear(in_features, out_features//3)
        self.linear_v = torch.nn.Linear(in_features, out_features//3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features//3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features//3].data

        self.linear_k.weight.data = target_layer.weight[target_layer.out_features//3:target_layer.out_features//3*2, :].data
        self.linear_k.bias.data = target_layer.bias[target_layer.out_features//3:target_layer.out_features//3*2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features//3*2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features//3*2:].data
    
    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q,k,v], dim = -1)


for key, module in model.named_modules():
    if key.endswith('attention'):
        try:
            # Here we split the query_key_value layer into three linear layer for LoRA. But you can also use merged linear.
            qkv_layer = QKV_layer(module.query_key_value.in_features, module.query_key_value.out_features) 
            qkv_layer.update(module.query_key_value)
            module.query_key_value = qkv_layer
        except:
            pass
        module.query_key_value = peft.tuners.lora.LoraModel(config, module.query_key_value)


lora.mark_only_lora_as_trainable(model)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
trainable_params = sum([np.prod(p.size()) for p in model_parameters])

model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
non_trainable_params = sum([np.prod(p.size()) for p in model_parameters]) 

print('trainable_params:{} ({:.2f}%)'.format(trainable_params, trainable_params/non_trainable_params*100,))

device = 'cuda'
EOS_ID = 150005

import json
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

with open('data/alpaca_data.json', 'r') as f:
    content = json.load(f)


pairs = []

for line in content:
    if line['input'] == '':
        prompt = PROMPT_DICT['prompt_no_input'].format_map(line)
    else:
        prompt = PROMPT_DICT['prompt_input'].format_map(line)
    completion = line['output']
    pairs.append({'prompt':prompt, 'completion':completion})


class AlpacaDataset(Dataset):
    def __init__(self, pairs, tokenizer) -> None:
        super().__init__()
        self.pairs = pairs
        self.tokenizer = tokenizer
 
    def __getitem__(self, index):
        prompt = self.tokenizer.encode(self.pairs[index]['prompt'])
        completion = self.tokenizer.encode(self.pairs[index]['completion'], add_special_tokens=False) + [EOS_ID]

        seq = prompt + completion
        context_length = seq.index(150004) + 1

        attention_mask = torch.ones((len(seq), len(seq)), device=device)
        attention_mask.tril_()
        attention_mask[..., :context_length - 1] = 1
        attention_mask.unsqueeze_(0)
        attention_mask = (attention_mask < 0.5).bool()

        position_ids = torch.stack([torch.arange(0,len(seq), device=device), torch.concat([torch.zeros(context_length-2, device=device), torch.arange(0,len(seq)-context_length+2, device=device)])]).long()
        labels = torch.tensor([-100] * len(prompt) + completion, device=device).long()

        return {'input_ids':seq, 'attention_mask':attention_mask, "labels":labels, 'position_ids':position_ids}
    
    def __len__(self):
        return len(self.pairs)


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    labels = []
    position_ids = []
    # TODO: padding for batch training
    for obj in batch:
        input_ids.append(obj['input_ids'])
        attention_mask.append(obj['attention_mask'])
        labels.append(obj['labels'])
        position_ids.append(obj['position_ids'])
    return {'input_ids': torch.tensor(input_ids).long(), 
            'attention_mask': torch.stack(attention_mask), 
            'labels': torch.stack(labels),
            'position_ids':torch.stack(position_ids)}

            

train_dataset = AlpacaDataset(pairs,tokenizer=tokenizer,)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn = collate_fn, shuffle=True, batch_size=1) 

from torch.cuda.amp import autocast
from transformers import get_linear_schedule_with_warmup

LR = 2e-5
NUM_EPOCHS = 2
accumulate_step = 32
version = 'test'

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
)


model.to(device).train()

with autocast(dtype=torch.float16):
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for step, batch in enumerate(t:=tqdm.tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss_d = outputs.loss.detach().float()
            t.set_description(f"loss: {loss_d}")
            total_loss += loss_d
            loss = outputs.loss / accumulate_step
            loss.backward()
            if (step+1) % accumulate_step == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

        peft_model_id = f"{ckpt_path}_{version}_{epoch}"
        torch.save(lora.lora_state_dict(model), peft_model_id+'.pt')
        print(epoch, total_loss/(step+1))
