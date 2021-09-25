import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from sklearn.model_selection import learning_curve, validation_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def plot_curves(estimator, title, X, y, axes=None, ylim=None, cv=None,
                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                n_neighbors=30):
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    # if axes is None:
    #     _, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    # Plot validation curves
    param_range = np.arange(1, n_neighbors)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name="n_neighbors", param_range=param_range,
        scoring="accuracy", n_jobs=n_jobs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # axes[3].set_title("Validation Curve")
    # axes[3].set_xlabel("Max Depth")
    # axes[3].set_ylabel("Score")

    # axes[3].grid()
    # axes[3].fill_between(param_range, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes[3].fill_between(param_range, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")
    # axes[3].plot(param_range, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # axes[3].plot(param_range, test_scores_mean, 'o-', color="g",
    #              label="Cross-validation score")
    # axes[3].legend(loc="best")

    axes[1].set_title("Validation Curve")
    axes[1].set_xlabel("N-Neighbors")
    axes[1].set_ylabel("Score")

    axes[1].grid()
    axes[1].fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[1].fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[1].plot(param_range, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[1].plot(param_range, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[1].legend(loc="best")

    return plt


if len(sys.argv) >= 2:

    dataset = int(sys.argv[1])
    if dataset == 0:
        df = pd.read_csv("datasets/tic-tac-toe/tic-tac-toe.data", header=None)
    elif dataset == 1:
        df = pd.read_csv(
            "datasets/credit-default/credit-default.data", header=None)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = X.apply(LabelEncoder().fit_transform)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# #############################################################################
# Fit regression model

    # n_neighbors = 8
    neighbors = []
    acc_uniform = []
    acc_distance = []

    max_acc = 0.0
    max_weight = ''
    max_n_neighbor = 0

    for j in range(1, 30):
        for i, weights in enumerate(['uniform', 'distance']):
            knn = KNeighborsClassifier(j, weights=weights)

            print('Processing Weight = ', weights)
            print('Processing K = ', j)
            tic = time.time()

            knn.fit(X_train, y_train)

            toc = time.time()
            print('Training Time Taken = ', toc - tic)

            y_pred = knn.predict(X_test)

            tic = time.time()
            print('Prediction Time Taken = ', tic - toc)

            acc_score = accuracy_score(y_test, y_pred)

            if acc_score > max_acc:
                max_acc = acc_score
                max_weight = weights
                max_n_neighbor = j

            if weights == 'uniform':
                acc_uniform.append(acc_score)
            else:
                acc_distance.append(acc_score)

        neighbors.append(j)

        # plt.subplot(2, 1, i + 1)
        # plt.scatter(X_train, y_train, color='darkorange', label='data')
        # plt.plot(X_test, pred, color='navy', label='prediction')
        # plt.plot(X_test, y_test, color='red', label='target')
        # plt.axis('tight')
        # plt.legend()
        # plt.title("KNeighborsRegressor (k = %i, weights = '%s')" %
        #           (n_neighbors, weights))
        # print('Weights = ', weights)
        # print('Accuracy = ', accuracy_score(y_test, y_pred))
        # print(classification_report(y_test, y_pred))

    print('Weight = ', max_weight)
    print('N-Neighbour = ', max_n_neighbor)
    print('Accuracy  = ', max_acc)

    d = pd.DataFrame({'acc_uniform': pd.Series(acc_uniform),
                      'acc_distance': pd.Series(acc_distance),
                      'neighbors': pd.Series(neighbors)})

    # visualizing changes in parameters
    plt.plot('neighbors', 'acc_uniform', data=d, label='uniform')
    plt.plot('neighbors', 'acc_distance', data=d, label='distance')
    plt.xlabel('n-neighbors')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    title = "Learning Curves"

    final_knn = KNeighborsClassifier(max_n_neighbor, weights=max_weight)
    plot_curves(final_knn, title, X, y,
                train_sizes=np.linspace(.1, 1.0, 9))

    plt.show()
    # # List Hyperparameters that we want to tune.
    # leaf_size = list(range(1, 50))
    # n_neighbors = list(range(1, 30))
    # p = [1, 2]
    # # Convert to dictionary
    # hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    # # Create new KNN object
    # knn_2 = KNeighborsClassifier()
    # # Use GridSearch
    # clf = GridSearchCV(knn_2, hyperparameters, cv=10)
    # # Fit the model
    # best_model = clf.fit(X_train, y_train)
    # # Print The value of best Hyperparameters
    # print('Best leaf_size:',
    #       best_model.best_estimator_.get_params()['leaf_size'])
    # print('Best p:', best_model.best_estimator_.get_params()['p'])
    # print('Best n_neighbors:',
    #       best_model.best_estimator_.get_params()['n_neighbors'])
    # plt.tight_layout()
    # plt.show()
