
from pathlib import Path
import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report

if __name__ == "__main__":
    #train = Path("data/train/features_train.csv")
    dev = Path("data/dev/features_dev.csv")
    #test = Path("data/test/features_test.csv")

    # read data
    #train = pd.read_csv(train, index_col=0, header = 0)
    dev = pd.read_csv(dev, index_col=0, header = 0)
    #test = pd.read_csv(test, index_col=0, header = 0)

    # specify features
    col_names = [feature for feature in dev.drop(["label", "original", "text"], axis=1)]

    # extract features
    y_dev = dev["label"].tolist()
    features = dev.drop(["label", "original", "text"], axis=1)
    X_dev = features.values.tolist()

    # initialize classifier
    clf = svm.SVC()
    clf.fit(X_dev, y_dev)

    # classify
    
    # NOTE: prep data only for program testing
    X_test = dev.copy()
    X_test = X_test.drop(["label", "original", "text"], axis=1)

    # predict
    y_test_pred = clf.predict(X_dev)
    X_test["label_pred"] = y_test_pred

    # save
    X_test.to_csv(Path("data/dev/pred.csv"))

    # analyse
    print(classification_report(y_dev, y_test_pred))
