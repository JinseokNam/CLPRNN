import time

import numpy as np
from scipy import sparse as sp

def list2matrix(A, stop_id, bos_id=0, n_labels=None, to_score=False, sparse=False):
    if n_labels is None:
        n_labels_ = 0
        for a in A:
            if n_labels_ < np.max(a):
                n_labels_ = np.max(a)
        n_labels = n_labels_

    n_samples = len(A)
    mat_dtype = np.float if to_score else np.int
    mat = sp.dok_matrix((n_samples, n_labels), dtype=mat_dtype)
    for idx in range(n_samples):
        for j, item in enumerate(A[idx]):
            if item == bos_id:
                continue
            try:
                if item == stop_id:
                    break
            except ValueError:
                print(item)
                sys.exit(1)

            # XXX item should be greater than 1 as 0 is BOS and 1 is EOS
            item -= 2       # offset both BOS and EOS

            if to_score:
                mat[idx, item] = 1 - (j / float(n_labels+1))
            else:
                mat[idx, item] = 1

    if sparse:
        return mat
    else:
        return mat.toarray()


def compute_exf1(predictions, targets, stop_id, **kwargs):
    """
    Parameters
    ----------
        predictions : List[ndarray]
        targets : List[ndarray]
        stop_id : int

    """

    batch_size = len(predictions)
    f1 = np.zeros(batch_size)

    for i in range(batch_size):
        stop_pos = np.argwhere(np.array(predictions[i][0]) == stop_id)
        stop_pos = stop_pos[0, 0] if stop_pos.shape[0] > 0 else 0

        t = set(targets[i][0]).difference(set([stop_id]))
        p = set(predictions[i][0][:stop_pos])      # ignore labels once a stop label has been generated

        inter_sec = t.intersection(p)

        pred_size = len(p)
        trg_size = len(t)
        correct_pred_size = len(inter_sec)

        if (pred_size == 0 and trg_size == 0) or correct_pred_size == 0:
            f1[i] = 0
        else:
            prec = correct_pred_size / pred_size if pred_size > 0 else 0
            recall = correct_pred_size / trg_size if trg_size > 0 else 0

            f1[i] = 2 * (prec * recall) / (prec + recall)

    return f1.mean()


def subset_accuracy(predictions, targets, **kwargs):

    result = np.all(targets == predictions, axis=1)
    result = np.mean(result, axis=0)

    return result


def compute_tp_fp_fn(predictions, targets, axis=0):
    # axis: axis for instance

    tp = np.sum(targets * predictions, axis=axis).astype('float32')
    fp = np.sum(np.logical_not(targets) * predictions, axis=axis).astype('float32')
    fn = np.sum(targets * np.logical_not(predictions), axis=axis).astype('float32')

    return (tp, fp, fn)


def safe_div(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
    return c[np.isfinite(c)]


def example_f1_score(predictions, targets, **kwargs):
    tp, fp, fn = compute_tp_fp_fn(predictions, targets, axis=1)
    example_f1 = safe_div(2*tp, 2*tp + fp + fn)

    f1 = np.mean(example_f1)

    return f1


def f1_score_from_stats(tp, fp, fn, average='micro'):
    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    if average not in set(['micro', 'macro']):
        raise ValueError("Specify micro or macro")

    if average == 'micro':
        f1 = 2*np.sum(tp) / \
            float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    elif average == 'macro':
        f1 = np.mean(safe_div(2*tp, 2*tp + fp + fn))

    return f1


def macro_f1(predictions, targets, **kwargs):
    tp, fp, fn = compute_tp_fp_fn(predictions, targets, axis=0)

    assert len(tp) == len(fp)
    assert len(fp) == len(fn)

    f1 = safe_div(2*tp, 2*tp + fp + fn)

    f1 = np.mean(f1)

    return f1


def micro_f1(predictions, targets, **kwargs):
    tp, fp, fn = compute_tp_fp_fn(predictions, targets, axis=0)

    assert len(tp) == len(fp)
    assert len(fp) == len(fn)
    f1 = 2*np.sum(tp) / \
        float(2*np.sum(tp) + np.sum(fp) + np.sum(fn))

    return f1


def precision_k(predictions, targets, K, average=True, **kwargs):
    batch_size, L = predictions.shape

    if average:
        P = np.zeros(len(np.arange(0, K, 2)))
    else:
        P = np.zeros((len(np.arange(0, K, 2)), batch_size))

    # corresponding to sort_sparse_mat
    j = np.argsort(predictions, axis=1)[:, ::-1]
    i = np.arange(batch_size)[:, np.newaxis]

    rank_mat = np.empty_like(j)
    rank_mat[i, j] = np.arange(L) + 1
    rank_mat[predictions == 0] = 0

    mat = np.empty_like(rank_mat)

    for i, k in enumerate(range(1, K+1, 2)):
        mat[:] = rank_mat
        mat[rank_mat > k] = 0
        mat[mat > 0] = 1
        mat = np.multiply(mat, targets, out=mat)
        num = np.sum(mat, axis=1)

        if average:
            P[i] = np.mean(num/float(k))
        else:
            P[i] = np.divide(num, float(k))

    return P


def nDCG_k(predictions, targets, K, average=True, **kwargs):
    batch_size, L = predictions.shape

    if average:
        P = np.zeros(len(np.arange(0, K, 2)))
    else:
        P = np.zeros((len(np.arange(0, K, 2)), batch_size))

    wts = 1/np.log2(np.arange(L) + 2)
    cum_wts = np.cumsum(wts)

    # corresponding to sort_sparse_mat
    j = np.argsort(predictions, axis=1)[:, ::-1]
    i = np.arange(batch_size)[:, np.newaxis]

    rank_mat = np.empty_like(j)
    rank_mat[i, j] = np.arange(L) + 1
    rank_mat[predictions == 0] = 0

    coeff_mat = np.empty_like(rank_mat, dtype=np.float)
    coeff_mat[:] = rank_mat
    mask = rank_mat != 0
    coeff_mat[mask] = 1./np.log2(coeff_mat[mask] + 1)
    mat = np.empty_like(coeff_mat, dtype=np.float)

    for i, k in enumerate(range(1, K+1, 2)):
        mat[:] = coeff_mat

        mat[rank_mat > k] = 0
        mat = np.multiply(mat, targets, out=mat)
        num = np.sum(mat, axis=1)

        count = np.sum(targets, axis=1, dtype=np.int)
        count = np.minimum(count, k)
        count[count == 0] = 1
        den = cum_wts[count-1]

        if average:
            P[i] = np.mean(num / den)
        else:
            P[i] = np.divide(num, den)

    return P
