import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from mrjob.job import MRJob

class IrisClassifierMRJOB(MRJob):

    def mapper(self, _, line):

        data = line.strip().split(",")
        features = [float(x) for x in data[:1]]
        label = data[-1]

        yield None, (label, *features)

    def reducer(self, _, label_features_pairs):
        df = pd.DataFrame(label_features_pairs, columns=["label","sepal_length","sepal_width","petal_length","petal_width"])
        X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        acc_score = accuracy_score(y_test, predictions)
        yield "Accuracy :",acc_score


if __name__ == "__main__":
    IrisClassifierMRJOB.run()
