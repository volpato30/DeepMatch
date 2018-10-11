import os
import pickle
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import config

logger = logging.getLogger(__name__)


@dataclass
class CandidatePeptide:
    peptide_str: str
    logp_score: str

class BaseParser(ABC):
    def __init__(self, mgf_file: str, feature_file: str):
        self.mgf_file = mgf_file
        self.feature_file = feature_file
        self.feature_location_list = []
        self.spectrum_location_dict = {}
        self.input_spectrum_handle = open(self.mgf_file, 'r')
        self.input_feature_handle = open(self.feature_file, 'r')
        self.get_location()

    def _parse_feature(self, feature_location):
        """read identified features and all candidate peptides"""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerIO: _parse_feature()")

        # self.input_feature_handle.seek(feature_location)
        # line = self.input_feature_handle.readline()
        # line = re.split(',|\r|\n', line)
        # feature_id = line[deepnovo_config.col_feature_id]
        # feature_area_str = line[deepnovo_config.col_feature_area]
        # feature_area = float(feature_area_str) if feature_area_str else 0.0
        # precursor_mz = float(line[deepnovo_config.col_precursor_mz])
        # precursor_charge = float(line[deepnovo_config.col_precursor_charge])
        # rt_mean = float(line[deepnovo_config.col_rt_mean])
        # raw_sequence = line[deepnovo_config.col_raw_sequence]
        # scan_list = re.split(';', line[deepnovo_config.col_scan_list])
        # ms1_list = re.split(';', line[deepnovo_config.col_ms1_list])
        # assert len(scan_list) == len(ms1_list), "Error: scan_list and ms1_list not matched."
        #
        # return feature_id, feature_area, precursor_mz, precursor_charge, rt_mean, raw_sequence, scan_list, ms1_list
        pass

    def _parse_spectrum_ion(self):
        """parse peaks in the mgf file"""

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerIO: _parse_spectrum_ion()")

        # ion
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > config.max_mz:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()

        return mz_list, intensity_list

    def get_location(self):
        """get feature location list for mgf files"""

        ### store file location of each spectrum for random access {scan:location}
        ### since mgf file can be rather big, cache the locations for each spectrum mgf file.
        spectrum_location_file = self.mgf_file + '.locations.pkl'
        if os.path.exists(spectrum_location_file):
            logger.info("read cached spectrum locations")
            with open(spectrum_location_file, 'rb') as fr:
                data = pickle.load(fr)
                self.spectrum_location_dict = data
        else:
            logger.info("build spectrum location from scratch")
            spectrum_location_dict = {}
            line = True
            while line:
                current_location = self.input_spectrum_handle.tell()
                line = self.input_spectrum_handle.readline()
                if "BEGIN IONS" in line:
                    spectrum_location = current_location
                elif "SCANS=" in line:
                    scan = re.split("=|\r|\n", line)[1]
                    spectrum_location_dict[scan] = spectrum_location

            self.spectrum_location_dict = spectrum_location_dict
            self.spectrum_count = len(spectrum_location_dict)
            logger.info(f"read {self.spectrum_count} spectrums from file {self.mgf_file}")

            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

        ### store location of each feature for random access
        feature_location_list = []
        # skip header line
        _ = self.input_feature_handle.readline()
        line = True
        while line:
            feature_location = self.input_feature_handle.tell()
            feature_location_list.append(feature_location)
            line = self.input_feature_handle.readline()
        feature_location_list = feature_location_list[:-1]  # the last line is EOS
        self.feature_location_list = feature_location_list
        logger.info(f"read {len(feature_location_list)} features")

    def get_feature(self, feature_index: int):
        pass


class TrainParser(BaseParser):
    """
    TrainParser will be responsible for reading training data, and write it into a tfrecord file.
    """
    def __init__(self, mgf_file: str, feature_file: str):
        super(TrainParser, self).__init__(mgf_file, feature_file)


class InferenceParser(BaseParser):
    """
    InferenceParser read in features, organize them in batches
    """
    def __init__(self, mgf_file: str, feature_file: str):
        super(InferenceParser, self).__init__(mgf_file, feature_file)
        self.batch_size = config.inference_batch_size

    def test_set_iterator(self):
        pass


def prepare_dataset_iterators(batch_size: int):
    pass
