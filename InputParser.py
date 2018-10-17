import os
import pickle
import re
import logging
from abc import ABC
import random
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
import config
import cython_func

logger = logging.getLogger(__name__)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


@dataclass
class CandidatePeptide:
    peptide_str: str
    logp_score: float
    is_decoy: bool = False


class BaseParser(ABC):
    def __init__(self, spectrum_file: str):
        self.spectrum_file = spectrum_file
        self.feature_location_list = []
        self.spectrum_location_dict = {}
        self.input_spectrum_handle = open(self.spectrum_file, 'r')
        self.get_location()

    @staticmethod
    def pad_to_length(x: list, length: int, pad_token):
        """
        change x inplace
        :param x:
        :param length:
        :param pad_token:
        :return:
        """
        for i in range(length - len(x)):
            x.append(pad_token)
        return x[:length]

    @staticmethod
    def get_peptide_features(peptide: list):
        """
        process the input peptide to the features used by model
        :param peptide: list of aa
        :return:
            peptide_sequence: list
            peptide_length: list
            peptide_location_index: int64 ndarray
        """
        peptide_sequence = [config.vocab[aa] for aa in peptide]
        peptide_length = [len(peptide_sequence)]
        peptide_location_index = []

        peptide_neutral_mass = sum([config.mass_ID[id] for id in peptide_sequence])
        prefix_mass = 0.0
        for id in peptide_sequence[:-1]:
            prefix_mass += config.mass_ID[id]
            ion_location_index = cython_func.get_ions_mz_index(peptide_neutral_mass, prefix_mass)
            peptide_location_index.append(ion_location_index)
        peptide_sequence = BaseParser.pad_to_length(peptide_sequence, config.peptide_max_length, config.PAD_ID)

        # pad location index with an invalid index so that it will be masked out in the computation graph.
        pad_ion_index = np.ones(config.num_ion_combination, dtype=np.int64) * (config.M + 1)
        peptide_location_index = BaseParser.pad_to_length(peptide_location_index,
                                                          config.peptide_max_length - 1,
                                                          pad_ion_index)
        peptide_location_index = np.array(peptide_location_index, dtype=np.int64)
        return peptide_sequence, peptide_length, peptide_location_index


    @staticmethod
    def parse_raw_sequence(raw_sequence: str):
        """

        :param raw_sequence:
        :return:
            peptide: a list of string, each string represent an aa
            unknown_modification: boolean, whether contain unknown modification or not.
        """
        raw_sequence_len = len(raw_sequence)
        peptide = []
        index = 0
        unknown_modification = False
        while index < raw_sequence_len:
            if raw_sequence[index] == "(":
                if peptide[-1] == 'M' and raw_sequence[index:index + 8] == "(+15.99)":
                    peptide[-1] = 'M(Oxidation)'
                    index += 8
                else:  # unknown modification
                    logger.error(f"encounter unknown modification in sequence {raw_sequence}")
                    unknown_modification = True
                    break
            else:
                peptide.append(raw_sequence[index])
                index += 1
        return peptide, unknown_modification

    def _parse_spectrum_ion(self):
        """parse peaks in the mgf file
        Returns:
            list of mz, list of intensities
        """

        # ~ print("".join(["="] * 80)) # section-separating line
        # ~ print("WorkerIO: _parse_spectrum_ion()")

        # ion
        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while line != '\n':
            mz, intensity = re.split('  |\r|\n', line)[:2]
            try:
                mz_float = float(mz)
                intensity_float = float(intensity)
            except ValueError as e:
                print("*" * 80)
                print(f"ERROR: got mz: {mz}, intensity: {intensity}")
                print("*" * 80)
                raise e

            # skip an ion if its mass > MZ_MAX
            if mz_float >= config.max_mz:
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
        spectrum_location_file = self.spectrum_file + '.locations.pkl'
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
                    scan = re.split("Scan ID: ", line)[1]
                    spectrum_location_dict[scan] = current_location

            self.spectrum_location_dict = spectrum_location_dict
            self.spectrum_count = len(spectrum_location_dict)
            logger.info(f"read {self.spectrum_count} spectrums from file {self.spectrum_file}")

            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)

    def get_scan(self, scan: str):
        """

        :param scan: scan id
        :return:
            mz_list
            intensity_list
            candidate_peptide_list: sorted by logp score in an descending order
        """
        self.input_spectrum_handle.seek(self.spectrum_location_dict[scan])
        candidate_peptides_list = []
        peptide_str = None
        peptide_logp_score = None
        is_decoy = None
        num_peaks = None
        line = self.input_spectrum_handle.readline()
        assert 'Scan ID' in line
        scan_id = line.strip().split('Scan ID: ')[1]
        for line in self.input_spectrum_handle:
            # loop through candidates
            if "Name:" in line:
                peptide_str = line.strip().split("Name: ")[1]
                peptide_str = peptide_str.split('/')[0]
            elif "Comment:" in line:
                # assuming there is alway a comment after a corresponding name.
                peptide_logp_score = float(line.strip('\n').split('Score= ')[1])
                accession = line.split('Accession=')[1].split(" Score=")[0]
                is_decoy = False if accession else True
                candidate_peptides_list.append(
                    CandidatePeptide(peptide_str=peptide_str,
                                     logp_score=peptide_logp_score,
                                     is_decoy=is_decoy)
                )
            if "Num Peaks:" in line:
                num_peaks = int(line.strip().split('Num Peaks: ')[1])
                break
        mz_list, intensity_list = self._parse_spectrum_ion()
        # sanity check
        if len(mz_list) != num_peaks:
            # it seems num_peaks in the original data is not correct
            logger.debug(f"scan:{scan_id} len_mz_list: {len(mz_list)} num_peaks: {num_peaks}")
        assert len(candidate_peptides_list) > 0
        # the first candidate should be the positive sample
        first_score = candidate_peptides_list[0].logp_score
        keyfunc = lambda x: x.logp_score
        sorted_candidate_list = sorted(candidate_peptides_list, key=keyfunc, reverse=True)
        # if first_score != sorted_candidate_list[0].logp_score:
        #     logger.warning(f"scan:{scan_id} the first candidate is not the one with highest score.")
        if len(sorted_candidate_list) > 1 + config.num_neg_candidates:
            logger.warning(f"scan:{scan_id} has {len(sorted_candidate_list)} candidates")
            sorted_candidate_list = sorted_candidate_list[:1 + config.num_neg_candidates]
        return mz_list, intensity_list, sorted_candidate_list


class TrainParser(BaseParser):
    """
    TrainParser will be responsible for reading training data, and write it into a tfrecord file.
    """

    def __init__(self, spectrum_file: str):
        super(TrainParser, self).__init__(spectrum_file)

    @staticmethod
    def make_example(mz_list, intensity_list, candidate_peptide_list):
        """

        :param mz_list:
        :param intensity_list:
        :param candidate_peptide_list: a list of CandidatePeptides, sorted by logp_score.
        :return:
            None if the pos_sample candidate peptides is longer than limit
            tf.train.Example object.
        """
        pos_sample = candidate_peptide_list[0]
        pos_peptide, unknown_mod = BaseParser.parse_raw_sequence(pos_sample.peptide_str)
        # get a random permutation of pos_peptide
        random_peptide = pos_peptide[:]
        random.shuffle(random_peptide)
        if unknown_mod:
            return None
        if len(pos_peptide) > config.peptide_max_length:
            logger.warning(f"skip pos sample {pos_sample.peptide_str} because of max length")
            return None
        spectrum_holder = cython_func.process_spectrum(mz_list, intensity_list)

        peptide_sequence, peptide_length, peptide_location_index = BaseParser.get_peptide_features(pos_peptide)

        input_spectrum_feature = tf.train.Feature(
            float_list=tf.train.FloatList(
                value=spectrum_holder.tolist()
            )
        )
        pos_peptide_sequence_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=peptide_sequence))
        pos_peptide_length_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=peptide_length))
        pos_peptide_location_index_feature = _bytes_feature(tf.compat.as_bytes(peptide_location_index.tostring()))

        neg_peptides = []
        for cp in candidate_peptide_list[1:]:
            neg_peptide, unknown_mod = BaseParser.parse_raw_sequence(cp.peptide_str)
            if unknown_mod:
                continue
            neg_peptides.append(neg_peptide)

        neg_peptides = BaseParser.pad_to_length(neg_peptides, config.num_neg_candidates, random_peptide)

        neg_peptide_sequence = []
        neg_peptide_length = []
        neg_peptide_location_index = []
        for neg_peptide in neg_peptides:
            peptide_sequence, peptide_length, peptide_location_index = BaseParser.get_peptide_features(neg_peptide)
            neg_peptide_sequence.append(peptide_sequence)
            neg_peptide_length.append(peptide_length)
            neg_peptide_location_index.append(peptide_location_index)
        neg_peptide_sequence = np.array(neg_peptide_sequence, np.int64)
        neg_peptide_length = np.array(neg_peptide_length, np.int64)
        neg_peptide_location_index = np.array(neg_peptide_location_index, np.int64)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "input_spectrum": input_spectrum_feature,
                    "pos_aa_sequence": pos_peptide_sequence_feature,
                    "pos_aa_sequence_length": pos_peptide_length_feature,
                    "pos_ion_location_index": pos_peptide_location_index_feature,
                    "neg_aa_sequence": _bytes_feature(tf.compat.as_bytes(neg_peptide_sequence.tostring())),
                    "neg_aa_sequence_length": _bytes_feature(tf.compat.as_bytes(neg_peptide_length.tostring())),
                    "neg_ion_location_index": _bytes_feature(tf.compat.as_bytes(neg_peptide_location_index.tostring())),
                }
            )
        )

        return example

    def convert_to_tfrecord(self, tfrecord_file_name: str, test_mode: bool=False):
        """

        :param tfrecord_file_name:
        :param test_mode: if in test mode, keep the scans order.(purely for unittest purpose)
        :return:
        """
        writer = tf.python_io.TFRecordWriter(tfrecord_file_name)
        scans = list(self.spectrum_location_dict.keys())
        if test_mode:
            pass
        else:
            # shuffle the scans list
            random.shuffle(scans)
        writed_scans = 0
        for i, scan in enumerate(scans):
            mz_list, intensity_list, candidate_peptide_list = self.get_scan(scan)
            example = self.make_example(mz_list, intensity_list, candidate_peptide_list)
            if example:
                writer.write(example.SerializeToString())
                writed_scans += 1
        writer.close()
        logging.info(f"in total {len(scans)} scans, write {writed_scans} to train set")


class InferenceParser(BaseParser):
    """
    InferenceParser read in features, organize them in batches
    """

    def __init__(self, spectrum_file: str):
        super(InferenceParser, self).__init__(spectrum_file)
        self.batch_size = config.inference_batch_size

    def test_set_iterator(self):
        pass


def tf_decode_reshape(x, dtype, shape: tuple):
    """
    helper function for decoder raw sequence and reshape
    :param x:
    :param shape:
    :return:
    """
    return tf.reshape(tf.decode_raw(x, dtype), shape)


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
        # numpy arrays stored as string
        "pos_ion_location_index": tf.FixedLenFeature([], tf.string),

        "neg_aa_sequence": tf.FixedLenFeature([], tf.string),
        "neg_aa_sequence_length": tf.FixedLenFeature([], tf.string),
        "neg_ion_location_index": tf.FixedLenFeature([], tf.string),
    }
    parsed_data = tf.parse_single_example(serialized=ex, features=read_features)

    return {
        "input_spectrum": parsed_data["input_spectrum"],
        "pos_aa_sequence": parsed_data["pos_aa_sequence"],
        "pos_aa_sequence_length": parsed_data["pos_aa_sequence_length"],
        "pos_ion_location_index": tf_decode_reshape(parsed_data["pos_ion_location_index"],
                                                    tf.int64,
                                                    (config.peptide_max_length - 1, config.num_ion_combination)
                                                    ),
        "neg_aa_sequence": tf_decode_reshape(parsed_data["neg_aa_sequence"],
                                             tf.int64,
                                             (config.num_neg_candidates, config.peptide_max_length)
                                             ),
        "neg_aa_sequence_length": tf_decode_reshape(parsed_data["neg_aa_sequence_length"],
                                                    tf.int64,
                                                    (config.num_neg_candidates, 1)
                                                    ),
        "neg_ion_location_index": tf_decode_reshape(parsed_data["neg_ion_location_index"],
                                                    tf.int64,
                                                    (config.num_neg_candidates,
                                                     config.peptide_max_length - 1,
                                                     config.num_ion_combination)
                                                    ),
    }


def select_neg(x: dict):
    """
    select the highest score neg sample.
    :param x:
    :return:
    """
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
