import numpy as np
from InputParser import InferenceParser, ScanPSM
import config
import logging
import logging.config


def _filter_by_fdr(scan_list: list, fdr: float, sort_key_func=lambda x: x.retrieved_peptide.logp_score):
    """
    input a list of Scan, output a subset of that list which passed the fdr control
    ******
    this function sort scan_list by logp_score inplace
    ******
    :param sort_key_func:
    :param scan_list:
    :param fdr:
    :return:
        filtered_scan_list
    """
    num_scans = len(scan_list)
    sorted_scan_list = sorted(scan_list, key=sort_key_func, reverse=True)
    filtered_list = []
    target_count, decoy_count = 0, 0
    for scan in sorted_scan_list:
        if not scan.retrieved_peptide.is_decoy:
            target_count += 1
            filtered_list.append(scan)
        else:
            decoy_count += 1
        if (decoy_count / (target_count + 1e-7)) > fdr:
            break
    logging.info(f"{target_count} out of {num_scans} PSMs satisfy fdr = {fdr}")
    return filtered_list


def nodup_split(scan_list: list, p):
    """
    split such that results do not overlap in peptides
    :param scan_list:
    :param p:
    :return:
        a list contain scan_list
    """
    peptide_to_scan_dict = {}
    for scan in scan_list:
        if scan.retrieved_peptide.peptide_str not in peptide_to_scan_dict:
            peptide_to_scan_dict[scan.retrieved_peptide.peptide_str] = [scan]
        else:
            peptide_to_scan_dict[scan.retrieved_peptide.peptide_str].append(scan)
    peptides = list(peptide_to_scan_dict.keys())
    result_list = [[] for i in range(len(p))]

    for peptide in peptides:
        ii = np.random.choice(list(range(len(p))), p=p)
        tmp_list = result_list[ii]
        tmp_list.extend(peptide_to_scan_dict[peptide])
    for tmp_list in result_list:
        if len(tmp_list) == 0:
            raise ValueError(f"too few datapoints for spliting with p={p}")
    return result_list


def write_scans(scan_list: list, read_file_handle, dest_file: str):
    """

    :param scan_list:
    :param read_file_handle: a file handle for reading
    :param dest_file: filename for output
    :return:
    """
    with open(dest_file, 'w') as fw:
        for scan in scan_list:
            prev_line = None
            cur_line = None
            read_file_handle.seek(scan.location_in_file)
            for line in read_file_handle:
                fw.write(line)
                prev_line = cur_line
                cur_line = line
                if prev_line == '\n' and cur_line == '\n':
                    break


if __name__ == "__main__":
    log_file_name = 'fdr_split.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)

    parser = InferenceParser("/home/rui/work/DeepMatch/data/hla_patient15_all_fractions/psm-training-mm15.txt")
    raw_scan_list = []
    for scan_id, location in parser.spectrum_location_dict.items():
        _, _, cp_list = parser.get_scan(scan_id)
        scan = ScanPSM(scan_id=scan_id, retrieved_peptide=cp_list[0], location_in_file=location)
        raw_scan_list.append(scan)
    train_scan_list, test_scan_list = nodup_split(raw_scan_list, p=np.array([0.95, 0.05]))
    write_scans(test_scan_list, parser.input_spectrum_handle, config.test_file)

    scan_list = _filter_by_fdr(train_scan_list, config.fdr_threshold)
    train_list, valid_list = nodup_split(scan_list, p=np.array([0.95, 0.05]))
    logging.info(f"train: {len(train_list)}\tvalid: {len(valid_list)}\ttest(not filtered): {len(test_scan_list)}")
    write_scans(train_list, parser.input_spectrum_handle, config.train_file)
    write_scans(valid_list, parser.input_spectrum_handle, config.valid_file)

