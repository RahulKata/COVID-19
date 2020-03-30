import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    df = pd.read_excel('symptoms.xls')

    y = df['Infection'].values
    x = df[['Age', 'Temperature', 'Cough',
            'Shortness of Breath', 'Headache', 'Body Pain']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    with open('model.pkl', 'wb') as file:
        pickle.dump(clf, file)
