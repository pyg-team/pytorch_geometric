import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    test_accuracies_all = []
    train_accuracies_all = []
    num_it_all = []

    for i in range(num_repetitions):
        # Train acc. over all folds.
        train_accuracies = []
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_feature_matrices[0]
            best_c = C[0]
            best_gi = 1

            for gi, gram_matrix in enumerate(all_feature_matrices):
                train = gram_matrix[train_index]
                val = gram_matrix[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = LinearSVC(C=c, tol=0.01, dual=True)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix
                        best_gi = gi + 1

            # Determine test accuracy.
            train = best_gram_matrix[train_index]
            test = best_gram_matrix[test_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = LinearSVC(C=best_c, tol=0.01, dual=False)
            clf.fit(train, c_train)
            train_acc = accuracy_score(c_train, clf.predict(train)) * 100.0

            gm = test

            y = clf.decision_function(gm)

            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            train_accuracies.append(train_acc)
            num_it_all.append(best_gi)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        train_accuracies_all.append(float(np.array(train_accuracies).mean()))

    return (
    round(np.array(train_accuracies_all).mean(), 1), round(np.array(train_accuracies_all).std(), 1), round(np.array(test_accuracies_all).mean(), 1),
    round(np.array(test_accuracies_all).std(), 1), round(np.array(num_it_all).mean(), 1))

# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, it, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]):
    # Acc. over all repetitions.
    test_accuracies_all = []
    train_accuracies_all = []
    num_it_all = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        train_accuracies = []
        kf = KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(list(range(len(classes)))):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1)
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]
            best_gi = 1

            for gi, gram_matrix in enumerate(all_matrices):
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix
                        best_gi = it[gi]

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0
            best_train = accuracy_score(c_train, clf.predict(train)) * 100.0

            test_accuracies.append(best_test)
            train_accuracies.append(best_train)
            num_it_all.append(best_gi)

        test_accuracies_all.append(float(np.array(test_accuracies).mean()))
        train_accuracies_all.append(float(np.array(train_accuracies).mean()))

    return (
    round(np.array(train_accuracies_all).mean(), 1), round(np.array(train_accuracies_all).std(), 1), round(np.array(test_accuracies_all).mean(), 1),
    round(np.array(test_accuracies_all).std(), 1), round(np.array(num_it_all).mean(), 1))

