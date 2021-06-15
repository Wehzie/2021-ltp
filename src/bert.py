from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
import os
import sys
from pathlib import Path
import torch

sys.path.append(os.getcwd())

from src.utils import make_bert_datasets

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

dev_path = Path("data/dev/europarl_dev.csv")

# read data
data = pd.read_csv(dev_path, index_col=0, header = 0, nrows=20) 

# format and split data
data = make_bert_datasets(data)
size = len(data)
train = data[:int(size*0.7)]
dev = data[int(size*0.7):int(size*0.85)]
test = data[int(size*0.85):]

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    "bert", "bert-base-cased", args=model_args, use_cuda=torch.cuda.is_available()
)

# Train the model
model.train_model(train)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev)

print(result)

# Make predictions with the model
# predictions, raw_outputs = model.predict(test['text'])