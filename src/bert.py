import argparse
import sys
import os
import logging
import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs

sys.path.append(os.getcwd())
from src.utils import make_bert_datasets

parser = argparse.ArgumentParser(description='Train and use Bert model.')
parser.add_argument('--nrows', default=60000, type=int, nargs='?',
                    help='number of data rows')
parser.add_argument('--epochs', default=1, type=int, nargs='?',
                    help='number of data rows')
parser.add_argument('--override', default=False, type=bool, nargs='?',
                    help='override model output directory')

args = parser.parse_args()
print(args)

def load_model_pth(model_path: Path):
    """
    Load a pytorch model from a model.pth; including weights and model structure.
    """
    model = torch.load(model_path)
    return model


model_args_list = [
    ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-5,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-5,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-4,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-4,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-5,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-5,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-4,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-4,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
        ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-6,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=2,
        learning_rate=4e-6,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
        ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-6,
        train_batch_size=8,
        overwrite_output_dir=args.override
    ),
    ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-6,
        train_batch_size=16,
        overwrite_output_dir=args.override
    ),
]

def get_trained(train, model_args):
    """
    train: train data
    return: trained model
    """
    # Create a ClassificationModel
    model = ClassificationModel(
        "bert", "bert-base-cased", args=model_args, use_cuda=torch.cuda.is_available()
    )

    # Train the model
    model.train_model(train)
    return model

def test_model(test, model):
    """
    test: test data
    model: trained model
    """
    # predict on test set
    X_test = test["text"].values.tolist()
    predictions, raw_outputs = model.predict(X_test)
    test.insert(1, "pred", predictions)    

    print(test.head())
    
    # save
    test.to_csv(Path("data/test_pred.csv"))

    # analyse
    print(classification_report(test["labels"], test["pred"]))

def get_accuracy(tp, fp, tn, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def grid_search(train, dev) -> None:
    """
    train: train data
    dev: dev data
    returns: None
    """
    df_grid = []
    for i, model_args in enumerate(model_args_list):
        model = get_trained(train, model_args)
        result, model_outputs, wrong_predictions = model.eval_model(dev)
        accuracy = get_accuracy(
            result["tp"],
            result["fp"],
            result["tn"],
            result["fn"],
        )
        print(model_args)
        print(result)
        print(accuracy)
        df_grid.append(str(model_args))
        df_grid.append(str(result))
        df_grid.append(str(accuracy))
        torch.save(model, f'data/model{i}.pth')
    
    with open('grid_search.txt', 'w') as f:
        f.write(json.dumps(df_grid))

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

    # train best model
    model_args = ClassificationArgs(
        num_train_epochs=3,
        learning_rate=4e-5,
        train_batch_size=16,
        overwrite_output_dir=args.override
    )
    model = get_trained(train, model_args)

    # test best model
    test_model(test, model)

    
    
