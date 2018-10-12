import os
import pickle
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
import tensorflow as tf
import config

logger = logging.getLogger(__name__)


@dataclass
class CandidatePeptide:
    peptide_str: str
    logp_score: str


class BaseParser(ABC):
    def __init__(self, mgf_file: str):
        self.mgf_file = mgf_file
        self.feature_location_list = []
        self.spectrum_location_dict = {}
        self.input_spectrum_handle = open(self.mgf_file, 'r')
        self.get_location()

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
                if "Scan ID:" in line:
                    spectrum_location = current_location
                    scan = re.split("Scan ID: ", line)[1]
                    spectrum_location_dict[scan] = current_location

            self.spectrum_location_dict = spectrum_location_dict
            self.spectrum_count = len(spectrum_location_dict)
            logger.info(f"read {self.spectrum_count} spectrums from file {self.mgf_file}")

            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

    def get_scan(self, scan: str):
        self.input_spectrum_handle.seek(self.spectrum_location_dict[scan])
        candidate_peptides_list = []
        for line in self.input_spectrum_handle:
            if "Name:" in line:
                peptide_str = re.split("Name: ", line)[1]
            elif "Comment:" in line:
                pass
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


def parse_tfrecord(ex):
    """
    parse tf.train.Example protocolbuf
    :param ex: serialized example
    :return:
    """
    read_features = {
        "input_spectrum": tf.FixedLenFeature([config.M], tf.float32),
        "pos_aa_sequence": tf.FixedLenFeature([config.peptide_max_length], tf.int64),
        "pos_aa_sequence_length": tf.FixedLenFeature([1], tf.int64),
        "pos_ion_location_index": tf.FixedLenFeature([config.peptide_max_length - 1, config.num_ion_combination],
                                                     tf.int64),

        "neg_aa_sequence": tf.FixedLenFeature([config.num_neg_candidates, config.peptide_max_length], tf.int64),
        "neg_aa_sequence_length": tf.FixedLenFeature([config.num_neg_candidates, 1], tf.int64),
        "neg_ion_location_index": tf.FixedLenFeature([config.num_neg_candidates,
                                                      config.peptide_max_length - 1,
                                                      config.num_ion_combination],
                                                     tf.int64),
    }
    parsed_data = tf.parse_single_example(serialized=ex, features=read_features)

    return parsed_data


def select_neg(x: dict):
    x['neg_aa_sequence'] = x['neg_aa_sequence'][0, :]
    x["neg_aa_sequence_length"] = x["neg_aa_sequence_length"][0, :]
    x["neg_ion_location_index"] = x["neg_ion_location_index"][0, :, :]
    return x


def make_dataset(file_path: str, batch_size: int, num_processes=config.num_processes):
    """
    read tfrecord from hard drive and return an tf.data.Dataset object
    :param file_path: path for tfrecord files
    :return:
    """
    dataset = tf.data.TFRecordDataset([file_path])
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=num_processes)
    dataset = dataset.map(select_neg, num_parallel_calls=num_processes)
    # Shuffle the dataset
    # dataset = dataset.shuffle(buffer_size=20)
    dataset = dataset.batch(batch_size)

    return dataset


def prepare_dataset_iterators(batch_size: int):
    """

    :param batch_size:
    :return:
    """
    train_ds = make_dataset(config.train_record_path, batch_size)
    valid_ds = make_dataset(config.valid_record_path, batch_size)

    iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    next_element = iterator.get_next()

    training_init_op = iterator.make_initializer(train_ds)
    validation_init_op = iterator.make_initializer(valid_ds)

    return next_element, training_init_op, validation_init_op
