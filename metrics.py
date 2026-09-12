"""Order macro F1 with the case-sensitive Kaggle None token."""
import numpy as np

def order_f1(actual, predicted):
    actual = set(map(str, actual)) or {'None'}
    predicted = set(map(str, predicted)) or {'None'}
    return 2 * len(actual & predicted) / (len(actual) + len(predicted))

def grouped_f1(group, truth, probability, n_orders, threshold, none_probability=None, none_threshold=1.1):
    """Every target order is counted, including zero true/predicted positives.

    group is a dense integer order index. truth is candidate binary label.
    Candidates must contain every true reordered product (audited upstream).
    """
    chosen = probability >= threshold
    true_n = np.bincount(group, weights=truth, minlength=n_orders)
    pred_n = np.bincount(group, weights=chosen, minlength=n_orders)
    tp = np.bincount(group, weights=truth * chosen, minlength=n_orders)
    pred_none = pred_n == 0
    if none_probability is not None:
        pred_none |= none_probability >= none_threshold
    true_none = true_n == 0
    return 2 * (tp + (true_none & pred_none)) / (true_n + true_none + pred_n + pred_none)
