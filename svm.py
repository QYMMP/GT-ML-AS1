import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.kernel_approximation import Nystroem
from sklearn.exceptions import ConvergenceWarning


def plot_curves(estimator, title, X, y, axes=None, ylim=None, cv=None,
                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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
    param_range = np.logspace(-6, 2, 5)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=ConvergenceWarning, module="sklearn")
        train_scores, test_scores = validation_curve(
            estimator, X, y, param_name="gamma", param_range=param_range,
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

    axes[1].set_title("Validation Curve with SVM")
    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.1)
    lw = 2
    axes[1].semilogx(param_range, train_scores_mean, label="Training score",
                     color="darkorange", lw=lw)
    axes[1].fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="darkorange", lw=lw)
    axes[1].semilogx(param_range, test_scores_mean,
                     label="Cross-validation score", color="navy", lw=lw)
    axes[1].fill_between(param_range, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="navy", lw=lw)
    axes[1].legend(loc="best")

    # axes[1].set_title("Validation Curve")
    # axes[1].set_xlabel("N-Neighbors")
    # axes[1].set_ylabel("Score")

    # axes[1].grid()
    # axes[1].fill_between(param_range, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes[1].fill_between(param_range, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")
    # axes[1].plot(param_range, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # axes[1].plot(param_range, test_scores_mean, 'o-', color="g",
    #              label="Cross-validation score")
    # axes[1].legend(loc="best")

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

    feature_map_nystroem = Nystroem(kernel='poly')
    X = feature_map_nystroem.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    max_acc = 0.0
    max_kernel = ''

    acc = []

    kernels = ['poly', 'rbf']
    for kernel in kernels:
        # svc = svm.LinearSVC(max_iter=100000)
        # print('Processing Kernel = Linear')
        # tic = time.time()

        # svc.fit(X_train, y_train)

        # toc = time.time()
        # print('Training Time Taken = ', toc - tic)

        # y_pred = svc.predict(X_test)

        # tic = time.time()
        # print('Prediction Time Taken = ', tic - toc)

        # acc_score = accuracy_score(y_test, y_pred)
        # acc.append(round(acc_score, 4))

        # max_acc = acc_score
        # max_kernel = 'Linear'

        svc = svm.SVC(kernel=kernel)
        print('Processing Kernel = ', kernel)
        tic = time.time()

        svc.fit(X_train, y_train)

        toc = time.time()
        print('Training Time Taken = ', toc - tic)

        y_pred = svc.predict(X_test)

        tic = time.time()
        print('Prediction Time Taken = ', tic - toc)

        acc_score = accuracy_score(y_test, y_pred)
        acc.append(round(acc_score, 4))

        if acc_score > max_acc:
            max_acc = acc_score
            max_kernel = kernel

    print('Kernel = ', max_kernel)
    print('Accuracy = ', max_acc)

    plt.bar(kernels, acc)
    for i, v in enumerate(acc):
        plt.text(i-0.15, v, str(v), color='blue')

    plt.show()

    final_svm = svm.SVC(kernel=max_kernel)

    title = "Learning Curves"

    plot_curves(final_svm, title, X, y, train_sizes=np.linspace(.1, 1.0, 9))

    plt.show()
