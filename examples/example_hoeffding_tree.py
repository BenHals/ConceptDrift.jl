from skmultiflow.data.sine_generator import SineGenerator
from skmultiflow.trees.hoeffding_tree import HoeffdingTree
import time

def get_classifier_accuracy_test(c, a):
    start = time.perf_counter()
    for i in range(1000000):
        X,y = a.next_sample()
        c.partial_fit(X, y)

    right = 0
    wrong = 0
    for i in range(100000):
        X,y = a.next_sample()
        label = c.predict(X)
        if label == y[0]:
            right += 1
        else:
            wrong += 1
    acc = right / (right + wrong)
    end = time.perf_counter()
    return acc, end-start

a = SineGenerator()
c = HoeffdingTree(split_criterion="gini", no_preprune=True, leaf_prediction="nb")
test_result = get_classifier_accuracy_test(c, a)
print(test_result)