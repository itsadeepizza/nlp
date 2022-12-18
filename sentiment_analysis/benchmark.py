from config import selected_config as conf
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
conf.set_derivate_parameters()

def build_matrices(dataset, used_tokens=None):
    X = []
    y = []
    for i in range(len(dataset)):
        tokenized, _, label = dataset[i]
        dummy_vector = np.zeros(conf.MAX_ID_TOKEN)
        dummy_label = np.zeros(4)
        for token in tokenized:
            dummy_vector[token] += 1
        # dummy_label[label] = 1
        X.append(dummy_vector)
        y.append(label)

    # remove columns of zeros, elsewhere the algorithm could not work
    X = np.stack(X)
    y = np.stack(y)
    if used_tokens is None:
        used_tokens = []
        for token in range(conf.MAX_ID_TOKEN):
            if X[:, token].sum() > 0:
                used_tokens.append(token)
    X = X[:, used_tokens]
    return used_tokens, X, y

if __name__ == '__main__':
    from loader import train, test
    from loader import dataset_train, dataset_test
    used_tokens, X_train, y_train = build_matrices(dataset_train)
    # Uncomment below to test using a random classifier
    # np.random.shuffle(y_train)
    _, X_test, y_test = build_matrices(dataset_test, used_tokens=used_tokens)
    # Create the logistic regression model
    means = X_train.mean(axis=0)
    X_train = X_train / means
    X_test = X_test / means


    # Logistic regression
    # model = LogisticRegression(solver='saga', multi_class='auto', penalty="l1", C=10)

    # Random Forest
    model = RandomForestClassifier(random_state=42, max_depth=None, min_samples_leaf=1, n_estimators=100)

    # Grid search on random forest
    #-----------------------
    # model_base = RandomForestClassifier(random_state=42)
    # from sklearn.model_selection import GridSearchCV
    #
    # # Create a parameter grid to search over
    # param_grid = {
    #     'n_estimators': [10, 50, 100, 200],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_leaf': [1, 2, 4]
    #     }
    #
    # # Create a grid search object using 5-fold cross-validation
    # grid_search = GridSearchCV(model_base, param_grid, cv=5)
    #
    # # Fit the grid search object to the training data
    # grid_search.fit(X_train, y_train)
    # results = grid_search.cv_results_
    # # Convert the results to a DataFrame
    # df = pd.DataFrame(results)
    # # Create a pivot table with the mean test scores
    # pivot = df.pivot_table(index='param_n_estimators', columns=['param_max_depth', 'param_min_samples_leaf'], values='mean_test_score')
    # print(pivot)
    #
    # # Get the best combination of parameters
    # best_params = grid_search.best_params_
    # print(f"Best parms are {best_params}")
    # model = RandomForestClassifier(random_state=42, **best_params)
    #-----------------------------------


    # Calculate expected accuracy
    ratio_classes = np.bincount(y_train)
    most_frequent_class = ratio_classes.argmax()
    only_frequent_expected_accuracy = ratio_classes.max() / ratio_classes.sum()
    random_expected_accuracy = ((ratio_classes / ratio_classes.sum())**2).sum()

    # Fit the model to the train data
    model.fit(X_train, y_train)

    # Report results
    # MFC = Most Frequent Class, a model who guess using only most frequent class
    print(f"Expected MFC mean accuracy {only_frequent_expected_accuracy:.3f}")
    print(f"Expected random mean accuracy {random_expected_accuracy:.3f}")
    print(f"MEAN ACCURACY ON TEST : {model.score(X_test, y_test):0.3f}")
    print(f"Mean accuracy on train : {model.score(X_train, y_train):0.3f}")
    print("-"*30)
    print()

    # Report most significant words
    # print("Most significant words for the model")
    # for label_idx in range(4):
    #     print()
    #     print("-------------------")
    #     anti_map_label = {idx: label for label, idx in conf.MAP_LABEL.items()}
    #     print(f"Label {anti_map_label[label_idx]} :")
    #     first_ten = np.argsort(-np.abs(model.coef_), axis=1)[label_idx, :20]
    #     for idx_coeff in first_ten:
    #         word = conf.TOKENIZER.decode(used_tokens[idx_coeff])
    #         print(f"{word}: {model.coef_[label_idx, idx_coeff]:0.5f}")
    # print()

    # So for a mean accuracy located in the following ranges:
    #     acc < 0.45        it is just guessing
    #     0.45 < acc < 0.7  it is just looking to which words are used, but not at the order
    #     acc > 0.7         It outperforms simpler models
    #     acc > 0.82        It outperforms UmBERTo https://aclanthology.org/2021.wassa-1.8.pdf
