import numpy as np
import logging
from scipy.optimize import basinhopping
from functools import partial
from InputParser import CandidatePeptideDM
logger = logging.getLogger(__name__)

def compute_indentification_under_fdr(scan_list: list, fdr: float, score_func):
    """
    input a list of Scan, output a subset of that list which passed the fdr control
    ******
    this function sort scan_list by logp_score inplace
    ******
    :param score_func:
    :param scan_list:
    :param fdr:
    :return:
        filtered_scan_list
    """
    num_scans = len(scan_list)
    psm_list = []

    for scan in scan_list:
        cp = max(scan.retrieved_peptides, key=score_func)
        psm_list.append((score_func(cp), cp.is_decoy))

    target_count, decoy_count = 0, 0
    sorted_psm_list = sorted(psm_list, key=lambda x: x[0], reverse=True)
    for tup in sorted_psm_list:
        if not tup[1]:
            target_count += 1
        else:
            decoy_count += 1
        if (decoy_count / (target_count + 1e-7)) > fdr:
            break
    return target_count, num_scans


def get_optimized_weights(scan_list: list) -> np.ndarray:
    """

    :param scan_list:
    :return:
    """
    fdr = 0.01
    x0 = np.array([1, 0])

    def score_func_(candidate_peptide: CandidatePeptideDM, x):
        temp = np.array([candidate_peptide.logp_score, candidate_peptide.deep_match_score])
        return np.sum(temp * x)

    def target_func(x):
        score_func = partial(score_func_, x=x)
        target, _ = compute_indentification_under_fdr(scan_list, fdr, score_func)
        return -target

    def print_fun(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))

    minimizer_kwargs = {"method": "BFGS"}
    ret = basinhopping(target_func, x0, minimizer_kwargs=minimizer_kwargs, niter=200, callback=print_fun)
    logger.info("global minimum: x = [%.4f, %.4f], f(x0) = %.4f" % (ret.x[0], ret.x[1], ret.fun))
    return ret.x
