import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.externals
from sklearn import tree
import joblib
from graphviz import Digraph
from graphviz import Source



def data_model_train():
    music_data = pd.read_csv("music.csv")

    # show column and row
    # print(df.shape)

    X = music_data.drop(columns=["genre"])
    y = music_data["genre"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    # prediction = model.predict([[21, 1], [22, 0]])
    prediction = model.predict(X_test)

    score = accuracy_score(y_test, prediction)

    print(score)


def result_prediction():
    music_data = pd.read_csv("music.csv")

    # show column and row
    # print(df.shape)

    # X = music_data.drop(columns=["genre"])
    # y = music_data["genre"]
    #
    #
    #
    # model = DecisionTreeClassifier()
    # model.fit(X, y)
    # joblib.dump(model, "music-recommender.joblib")
    model = joblib.load("music-recommender.joblib")
    prediction = model.predict([[21, 1]])
    print(prediction)

def visualize_model():
    music_data = pd.read_csv("music.csv")


    X = music_data.drop(columns=["genre"])
    y = music_data["genre"]



    model = DecisionTreeClassifier()
    model.fit(X, y)

    tree.export_graphviz(model, out_file="music-recommender.dot", feature_names=["age", "gender"], class_names=sorted(y.unique()),
                         label="all", rounded=True, filled=True)



def main():
    visualize_model()
    # result_prediction()
    # data_model_train()


if __name__ == '__main__':
    main()
