from config import *
from pynvml import *

# utilization functions

def gpu_utilization():
  nvmlInit()
  handle = nvmlDeviceGetHandleByIndex(0)
  info = nvmlDeviceGetMemoryInfo(handle)
  print(f"Used GPU memory so far: {info.used//1024**2} MB")


gpu_utilization()

# setting up data

import re
from string import digits, punctuation
remove_digits = str.maketrans('', '', digits)
remove_punc = str.maketrans('','', punctuation)

'''
  remove punctuation
  remove digits
  remove statistical terms
  normalize whitespace
'''
def clean(row):
  text = row.full_text
  #text = text.translate(remove_punc)
  text = text.translate(remove_digits)
  stats = ["p","r","P","rho","ρ","Χ","ω","f","F","n","N","β","CI"]

  for stat in stats:
    expr = re.compile(r'\b' + re.escape(stat) + r'\b', re.IGNORECASE)
    text = expr.sub("", text)

  signs = ["(",")","<",">","=","±",".,","..", "  "]
  for sign in signs:
    text = text.replace(sign, "")
  
  return text#" ".join(text.split())


##

import pandas as pd

# note that the sample csvs can contain multiple replications for a study
train_df = pd.read_csv(working_path + "Data/training_sample.csv")
train_texts = pd.read_csv(working_path + "Data/dataset_training.csv")

# note that training set holds 388 papers, but only 348 unique dois (because multiple attempts at same study)
# which means training set actually = 348 data points
# we don't need to store the text multiple times ...
print("unique training data, n=", len(train_texts['doi'].unique()))

# todo: count these per doi, and take majority vote as training label
def compute_label(row):
  doi = row.doi
  replications = train_df[train_df['doi'] == doi]
  false_count = len(replications[replications['replicated_binary'] == 'no'])
  true_count = len(replications[replications['replicated_binary'] == 'yes'])

  if false_count + true_count == 0:
    return "error"
  if true_count == false_count:
    return "maybe"
  elif true_count > false_count:
    return "yes"
  else:
    return "no"

train_texts['label'] = train_texts.apply(compute_label, axis=1)
train_texts['full_text'] = train_texts.apply(clean, axis=1)


####
data = train_texts[train_texts['label'].isin(["yes", "no"])] # binary replication

Xs = data['full_text']
ys = data['label']

print("count", len(Xs))
print("replicated", 100.0 * (ys.where(ys == "yes").count()/len(Xs)), "%")

data.to_csv(working_path + "/Data/full_training_data.csv", index=False)


# Modeling

model_name = "domenicrosati/deberta-v3-large-dapt-tapt-scientific-papers-pubmed-finetuned-DAGPap22"

####


from transformers import AutoTokenizer, AutoModel, pipeline

tokenizer = AutoTokenizer.from_pretrained(model_name)

######


from datasets import Dataset

def yesNoToInt(row):
  y = row.label
  return int(y == "yes")

data["labels"] = data.apply(yesNoToInt, axis=1)
data["labels"] = pd.to_numeric(data["labels"])

dataset = Dataset.from_pandas(data).shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.15)
dataset = dataset.rename_column("full_text", "text")
dataset = dataset.remove_columns(["id", "doi", "label"])

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_papers = dataset.map(preprocess_function, batched=True)


####

from transformers import DataCollatorWithPadding
import evaluate
import numpy as np

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "no", 1: "yes"}
label2id = {"no": 0, "yes": 1}


###

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
)

###

gpu_utilization()

training_args = TrainingArguments(
    output_dir="deberta-replication",
    learning_rate=5e-7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=100,
    eval_steps=100,
    weight_decay=0.0005,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_papers["train"],
    eval_dataset=tokenized_papers["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


####

gpu_utilization()

trainer.train()
