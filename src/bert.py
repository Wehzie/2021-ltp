import argparse
import sys
import os
import logging
from pathlib import Path

import pandas as pd
import torch
from simpletransformers.classification import ClassificationModel, ClassificationArgs

sys.path.append(os.getcwd())
from src.utils import make_bert_datasets

parser = argparse.ArgumentParser(description='Train and use Bert model.')
parser.add_argument('--nrows', default=None, type=int, nargs='?',
                    help='number of data rows')
parser.add_argument('--epochs', default=1, type=int, nargs='?',
                    help='number of data rows')

args = parser.parse_args()
print(args)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    dev_path = Path("data/dev/europarl_dev.csv")

    # read data
    data = pd.read_csv(dev_path, index_col=0, header = 0, nrows=args.nrows) 

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
