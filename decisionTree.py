import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import printCurve as pc
import time

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve, validation_curve


def plot_curves(estimator, title, X, y, axes=None, ylim=None, cv=None,
                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                max_depth=30):
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
    param_range = np.arange(1, max_depth)
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name="max_depth", param_range=param_range,
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
    axes[1].set_xlabel("Max Depth")
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

    max_acc = 0.0
    max_acc_criterion = ''
    max_acc_depth = 0

    max_depth = []
    acc_gini = []
    acc_entropy = []
    for i in range(1, 30):
        dtree = DecisionTreeClassifier(criterion='gini', max_depth=i)
        print('Processing Criterion = Gini')
        print('Processing Depth = ', i)
        tic = time.time()

        dtree.fit(X_train, y_train)

        toc = time.time()
        print('Training Time Taken = ', toc - tic)

        pred = dtree.predict(X_test)

        tic = time.time()
        print('Prediction Time Taken = ', tic - toc)

        acc_score = accuracy_score(y_test, pred)
        if acc_score > max_acc:
            max_acc = acc_score
            max_acc_criterion = 'gini'
            max_acc_depth = i
        acc_gini.append(acc_score)

        ####
        dtree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
        print('Processing Criterion = Entropy')
        print('Processing Depth = ', i)
        tic = time.time()

        dtree.fit(X_train, y_train)

        toc = time.time()
        print('Training Time Taken = ', toc - tic)

        pred = dtree.predict(X_test)

        tic = time.time()
        print('Prediction Time Taken = ', tic - toc)

        acc_score = accuracy_score(y_test, pred)
        if acc_score > max_acc:
            max_acc = acc_score
            max_acc_criterion = 'entropy'
            max_acc_depth = i
        acc_entropy.append(acc_score)

        ####
        max_depth.append(i)

    d = pd.DataFrame({'acc_gini': pd.Series(acc_gini),
                      'acc_entropy': pd.Series(acc_entropy),
                      'max_depth': pd.Series(max_depth)})

    # visualizing changes in parameters
    plt.plot('max_depth', 'acc_gini', data=d, label='gini')
    plt.plot('max_depth', 'acc_entropy', data=d, label='entropy')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.legend()

    final_dtree = DecisionTreeClassifier(
        criterion=max_acc_criterion, max_depth=max_acc_depth)
    final_dtree.fit(X_train, y_train)
    pred = final_dtree.predict(X_test)
    print('Criterion = ', max_acc_criterion)
    print('Max Depth = ', max_acc_depth)
    print('Accuracy  = ', accuracy_score(y_test, pred))
    plt.show()

    # dot_data = StringIO()
    # export_graphviz(final_dtree, out_file=dot_data)
    # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_png('tree.png')
    # Image(graph.create_png())

    title = "Learning Curves"

    plot_curves(final_dtree, title, X, y,
                train_sizes=np.linspace(.1, 1.0, 9))

    plt.show()
