from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report

# sklearn uses numpy's random number generator
np.random.seed(50)

def normalize_features(df):
    """
    Take dataframe and normalize each feature.
    """
    # split text data and features
    meta = df[["label", "original", "text"]]
    features = df.drop(["label", "original", "text"], axis=1)

    # branch_index_all can be negative
    features["branching_index_all"] = features["branching_index_all"].add(
        abs(features["branching_index_all"].min(axis=0)))
    # normalize
    features = features.div(features.max(axis=0), axis=1)

    # combine text data and features
    full = pd.concat([meta, features], axis=1)
    return full

def remove_shorts(df):
    """
    Remove rows where sentences have 3 or less words in the original transcription.
    """
    df = df[df["original"].map(lambda x: len(x.split(" ")) > 3)]
    return df

if __name__ == "__main__":
    #train_path = Path("data/train/features_train.csv")
    dev_path = Path("data/dev/features_dev_t.csv")
    #test_path = Path("data/test/europarl_test.csv")

    # read data
    full = pd.read_csv(dev_path, index_col=0, header = 0, nrows=60000)  # TODO: all data
    
    # preprocess: remove short sentences
    full = remove_shorts(full)

    # preprocess: normalize each feature column
    full = normalize_features(full)

    # 70/15/15 train/dev/test split
    train = full[:int(len(full)*0.7)]
    dev = full[int(len(full)*0.7):int(len(full)*0.7 + len(full)*0.15)]
    test = full[int(len(full)*0.7 + len(full)*0.15):]

    # extract labels
    y_train = train["label"].tolist()
    y_dev = dev["label"].tolist()
    y_test = test["label"].tolist()
        
    # select best features
    train_features = train.drop(["label", "original", "text"], axis=1)
    dev_features = dev.drop(["label", "original", "text"], axis=1)
    test_features = test.drop(["label", "original", "text"], axis=1)
    
    selector = SelectKBest(chi2, k=8).fit(train_features, y_train)
    keep_cols = selector.get_support(indices=True)

    train_features = train_features.iloc[:,keep_cols]
    dev_features = dev_features.iloc[:,keep_cols]
    test_features = test_features.iloc[:,keep_cols]

    # convert features to lists for sklearn
    X_train = train_features.values.tolist()
    X_dev = dev_features.values.tolist()
    X_test = test_features.values.tolist()

    print("Preprocessing complete.")

    # initialize classifier
    clf = svm.SVC(kernel="linear")
    clf.fit(X_train, y_train)
    print("Classifier initialized.")

    # predict
    y_dev_pred = clf.predict(X_dev)
    dev.insert(1, "label_pred", y_dev_pred)    
    print("Predictions complete.")

    # save
    dev.to_csv(Path("data/dev/test_pred.csv"))

    # analyse
    print(classification_report(y_dev, y_dev_pred))

