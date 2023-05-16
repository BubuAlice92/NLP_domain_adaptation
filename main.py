### Domain adaptation: fine-tune a sentence-transformers model for medical use

# IMPORT ALL LIBRAIRIES
import multiprocessing
import pandas as pd
import numpy as np
import math
import glob
import torch
import matplotlib.pyplot as plt
import transformers

import sklearn
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoConfig
from transformers import BertForMaskedLM, DistilBertForMaskedLM
from transformers import BertTokenizer, DistilBertTokenizer
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from tokenizers import BertWordPieceTokenizer

# SET HYPERPARAMS
SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 2e-5 
LR_WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01


# LOAD, SPLIT AND TRANSFORM CSV INTO A DATASET OBJECT
# load data
dtf_mlm = pd.read_csv('fake.csv')

# Train/Valid Split
df_train, df_valid = train_test_split(
    dtf_mlm, test_size=0.15, random_state=SEED_SPLIT
)

len(df_train), len(df_valid)

# Convert to Dataset object
train_dataset = Dataset.from_pandas(df_train[['text']].dropna())
valid_dataset = Dataset.from_pandas(df_valid[['text']].dropna())


# LOAD MODEL AND TOKENIZER
MODEL = 'bert'
bert_type = 'bert-base-cased'

if MODEL == 'distilbert':
    TokenizerClass = DistilBertTokenizer 
    ModelClass = DistilBertForMaskedLM 
elif MODEL == 'bert':
    TokenizerClass = BertTokenizer
    ModelClass = BertForMaskedLM 
elif MODEL == 'roberta':
    TokenizerClass = RobertaTokenizer
    ModelClass = RobertaForMaskedLM
elif MODEL == 'scibert':
    TokenizerClass = AutoTokenizer
    ModelClass = AutoModelForMaskedLM


tokenizer = TokenizerClass.from_pretrained(
            bert_type, use_fast=True, do_lower_case=False, max_len=MAX_SEQ_LEN
            )
model = ModelClass.from_pretrained(bert_type)


# TOKENIZE DATASET
def tokenize_function(row):
    return tokenizer(
        row['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_special_tokens_mask=True)
  
column_names = train_dataset.column_names

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names,
)

valid_dataset = valid_dataset.map(
    tokenize_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=column_names,
)


# TRAIN THE MODEL
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


steps_per_epoch = int(len(train_dataset) / TRAIN_BATCH_SIZE)

training_args = TrainingArguments(
    output_dir='./bert-news',
    logging_dir='./LMlogs',             
    num_train_epochs=2,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    save_steps=steps_per_epoch,
    save_total_limit=3,
    weight_decay=WEIGHT_DECAY,
    learning_rate=LEARNING_RATE, 
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='loss', 
    greater_is_better=False,
    seed=SEED_TRAIN
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("model") #save your custom model


# PERPLEXICITY EVALUATION OF THE RAW MODEL
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast = False, do_lower_case=True)
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')

trainer = Trainer(
  model=model,
  data_collator=data_collator,
  #train_dataset=tokenized_dataset_2['train'],
  eval_dataset=valid_dataset,
  tokenizer=tokenizer,
  )

eval_results = trainer.evaluate()

print('Evaluation results: ', eval_results)
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")
print('----------------\n')


# PERPLEXICITY EVALUATION OF THE FINE-TUNED MODEL
for modelpath in glob.iglob("model"):
  print('Model: ', modelpath)
  tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False, do_lower_case=True)
  model = AutoModelForMaskedLM.from_pretrained(modelpath)

  trainer = Trainer(
    model=model,
    data_collator=data_collator,
    #train_dataset=tokenized_dataset_2['train'],
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    )
  
  eval_results = trainer.evaluate()

  print('Evaluation results: ', eval_results)
  print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")
  print('----------------\n')