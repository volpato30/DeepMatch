import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_integer("A",
                            128,
                            "")
tf.app.flags.DEFINE_integer("F",
                            256,
                            "")
tf.app.flags.DEFINE_integer("H",
                            4,
                            "")
tf.app.flags.DEFINE_float("init_lr",
                          1e-3,
                          "initial learning rate")
tf.app.flags.DEFINE_string("optimizer",
                           "adam",
                           "optimizer to use.")
tf.app.flags.DEFINE_string("activation",
                           "relu",
                           "activatino to use")
tf.app.flags.DEFINE_boolean("linear_cosine",
                            True,
                            "use linear cosine lr decay")
tf.app.flags.DEFINE_string("mode",
                           "train",
                           "mode for the main.py")

FLAGS = tf.app.flags.FLAGS

mode = FLAGS.mode

peptide_max_length = 15
num_ion_combination = 18
# M should be dividable by 32
M = 4000  # contrast to the original paper, we let the discretized spectrum start from 0
delta_M = 0.5
resolution = 2
max_mz = int(np.ceil(M * delta_M))

assert delta_M * resolution == 1

embed_dimension = FLAGS.A  # A in the original paper
lstm_output_dimension = FLAGS.F // 2  # F / 2 in the original paper
spectral_hidden_dimension = FLAGS.H  # H in the original paper

weight_decay = 1e-6

_PAD = "_PAD"
_START_VOCAB = [_PAD]

PAD_ID = 0

vocab_reverse = ['A',
                 'R',
                 'N',
                 #'N(Deamidation)',
                 'D',
                 'C',
                 #'C(Carbamidomethylation)',
                 'E',
                 'Q',
                 #'Q(Deamidation)',
                 'G',
                 'H',
                 'I',
                 'L',
                 'K',
                 'M',
                 'M(Oxidation)',
                 'F',
                 'P',
                 'S',
                 'T',
                 'W',
                 'Y',
                 'V',
                ]
vocab_reverse = _START_VOCAB + vocab_reverse
print("vocab_reverse ", vocab_reverse)

vocab = dict([(x, y) for (y, x) in enumerate(vocab_reverse)])
print("vocab ", vocab)

vocab_size = len(vocab_reverse)
print("vocab_size ", vocab_size)

mass_H = 1.0078
mass_H2O = 18.0106
mass_NH3 = 17.0265
mass_N_terminus = 1.0078
mass_C_terminus = 17.0027
mass_CO = 27.9949

mass_AA = {'_PAD': 0.0,
           #'_GO': mass_N_terminus-mass_H,
           #'_EOS': mass_C_terminus+mass_H,
           'A': 71.03711, # 0
           'R': 156.10111, # 1
           'N': 114.04293, # 2
           #'N(Deamidation)': 115.02695,
           'D': 115.02694, # 3
           'C': 103.00919, # 4
           # 'C(Carbamidomethylation)': 160.03065, # C(+57.02)
           #~ 'C(Carbamidomethylation)': 161.01919, # C(+58.01) # orbi
           'E': 129.04259, # 5
           'Q': 128.05858, # 6
           #'Q(Deamidation)': 129.0426,
           'G': 57.02146, # 7
           'H': 137.05891, # 8
           'I': 113.08406, # 9
           'L': 113.08406, # 10
           'K': 128.09496, # 11
           'M': 131.04049, # 12
           'M(Oxidation)': 147.0354,
           'F': 147.06841, # 13
           'P': 97.05276, # 14
           'S': 87.03203, # 15
           'T': 101.04768, # 16
           'W': 186.07931, # 17
           'Y': 163.06333, # 18
           'V': 99.06841, # 19
          }

mass_ID = [mass_AA[vocab_reverse[x]] for x in range(vocab_size)]
mass_AA_min = mass_AA["G"] # 57.02146

save_dir = './chkpoint'
num_epochs = 50

num_neg_candidates = 4

batch_size = 16
inference_batch_size = 32
summary_steps = 50

grad_clip = 1.0

fdr_threshold = 0.005
# train, valid, test file path
train_file = './data/train_scans.txt'
valid_file = './data/valid_scans.txt'
test_file = './data/test_scans.txt'

# tfrecord path
train_record_path = f'./data/train_M{M}.tfrecord'
valid_record_path = f'./data/valid_M{M}.tfrecord'

# piecewise constant learn rate
boundaries = [500000]
values = [FLAGS.init_lr, 1e-4]
# linear cosine learn rate decay
init_lr = FLAGS.init_lr
lr_decay_steps = num_epochs // 2 * 300000 // batch_size

keep_prob = 0.8

num_processes = 10

neg_sample_score_threshold = 0.95
histo_summary_flag = False