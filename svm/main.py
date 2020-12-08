import numpy as np
import auxiliarymethods.auxiliary_methods as aux
import auxiliarymethods.datasets as dp
import auxiliarymethods.weisfeiler_leman as wl
from auxiliarymethods.kernel_evaluation import linear_svm_evaluation, kernel_svm_evaluation



def main():

    # Load dataset.
    dataset = "ENZYMES"
    use_labels = True
    num_reps = 10

    dp.get_dataset(dataset)
    graph_db, classes = dp.read_txt(dataset)

    all_matrices = []
    for i in range(1, 6):
        gm = wl.compute_wl_1(graph_db, i, gram_matrix=False, uniform=not use_labels)
        gm_n = aux.normalize_feature_vector(gm)
        all_matrices.append(gm_n)
    acc_train, s_train, acc_test, s_test = linear_svm_evaluation(all_matrices, np.array(classes),
                                                                 num_repetitions=num_reps)

    print("WL1 " + str(acc_train) + " " + str(s_train) + " " + str(acc_test) + " " + str(s_test))
