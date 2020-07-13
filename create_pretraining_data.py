# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
# coding=utf-8
"""Create masked LM/next sentence masked_lm TF examples for ALBERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import random
from .tokenization import FastaTokenizer
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")

flags.DEFINE_string("spm_model_file", None,
                    "The model file for sentence piece tokenization.")

flags.DEFINE_string("input_file_mode", "r",
                    "The data format of the input file.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", True,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_bool(
    "do_permutation", False,
    "Whether to do the permutation training.")

flags.DEFINE_bool(
    "favor_shorter_ngram", True,
    "Whether to set higher probabilities for sampling shorter ngrams.")

flags.DEFINE_bool(
    "random_next_sentence", False,
    "Whether to use the sentence that's right before the current sentence "
    "as the negative sample for next sentence prection, rather than using "
    "sentences from other random documents.")

flags.DEFINE_integer("max_seq_length", 512, "Maximum sequence length.")

flags.DEFINE_integer("ngram", 3, "Maximum number of ngrams to mask.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 40,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")



def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, input_ids, input_mask, segment_ids, masked_lm_positions,
                masked_lm_ids, masked_lm_weights, next_sentence_labels):
    features = collections.OrderedDict()
    #The token IDs
    features["input_ids"] = create_int_feature(input_ids)
    
    #List of all 1's
    #https://github.com/google-research/albert/blob/4135001fb1a84491bb86bebb2646523495d36f74/modeling.py#L191
    features["input_mask"] = create_int_feature(input_mask)
    
    # Segment ID is used in bert to seperate sentences pairs. 
    # All sentence 1 tokens get segment ID 0, and all sentence 2 tokens get segment ID 1
    features["segment_ids"] = create_int_feature(segment_ids)
    
    #A list of the indexes of places that were masked
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    
    #An ordered list of Id's of the tokens that were masked 
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    

    # As far as I can tell, this isn't being used anywhere. Though, run_training.py does look for it (removed)
    #features["next_sentence_labels"] = create_int_feature([0])

    self.tf_example = tf.train.Example(features=tf.train.Features(feature=features))


  def serialize(self):
    return self.tf_example.SerializeToString()



class TrainingExmpleWriter():
  """
    Data writer. It will round-robin write to the output file paths in TFRecord format
    Record will be stored in buffer until threshold is reached before writing 
    for performance on large files
    Buffer Size is number of records

    **IMPORTANT**
    Must call flush_and_close() when finished to make sure all data is written to disk
  """
  def __init__(self, output_files: [str]):
    self.writers = self._create_output_writers(output_files)
    self.writer_index = 0
    self.total_written = 0

  def flush_and_close(self):
    for writer in output_writers_queue:
      writer.flush()
      writer.close()


  def write(self, trainingInstance: TrainingInstance):
    self.writers[writer_index].write(trainingInstance.serialize())

    self.writer_index = (self.writer_index + 1) % len(self.writers)

    total_written += 1


  def total_written(self):
    return self.total_written


  def _create_output_writers(self, output_files: [str]):
    output_file_writers = []
    tf.logging.info("*** Writing output files ***")
    for output_file in output_files:
      tf.logging.info("  %s", output_file)
      output_file_writers.append(tf.python_io.TFRecordWriter(output_file))
    
    return output_file_writers






def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  #Rewritten for single sequence usecases without next sentence prediciton support
  #This should greatly simplify this function over the one that comes with albert
  #Some of the code comes from https://github.com/lonePatient/albert_pytorch/blob/master/prepare_lm_data_mask.py

  num_to_mask = min(max_predictions_per_seq, 
                    max(1, int(round(len(tokens) * masked_lm_prob))))

  cand_indexes = range(len(tokens))
  mask_indexes = sorted(rng.sample(cand_indexes, num_to_mask))

  masked_token_labels = []
  for idx in mask_indexes:
    masked_token = None
    if rng.random() < 0.8:
      masked_token = '[MASK]'
    else:
      # 10% of the time, keep original
      if rng.random() < 0.5:
        masked_token = tokens[idx]
      else:
        masked_token = rng.choice(vocab_words)

    masked_token_labels.append(MaskedLmInstance(index=idx, label=tokens[idx]))
    tokens[index] = masked_token
  
  assert len(masked_token_labels) <= num_to_mask

  masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)
  mask_indices = [p.index for p in masked_token_labels]
  masked_labels = [p.label for p in masked_token_labels]
  
  return tokens, mask_indices, masked_labels



def process_input_files(input_files: [str], tokenizer: FastaTokenizer, 
                          dataWriter: TrainingExmpleWriter, rng):
  """
    Process over entire data set
    Steps for each row in each file:
      Tokenize
      Limit sequence length
      Create masks
      Create a TrainingInstance
      Write the training instance

  """
  for input_file in input_files:
    process_input_file(input_file, tokenizer, dataWriter)


def process_input_file(input_file: str, tokenizer: FastaTokenizer, 
                        dataWriter: TrainingExmpleWriter, rng):
  """
    process a single file
  """
  with tf.gfile.GFile(input_file, FLAGS.input_file_mode) as reader:
    while True:
      line = reader.readline()
      if not line:
        break
      
      tokens = tokenizer.tokenize(line)

      max_seq_length = FLAGS.max_seq_length 
      if len(tokens) > max_seq_length:
        # 50% probability of starting from the left or right
        # This is to reduce bias from training data
        if rng.random() < 0.5:
          tokens = tokens[:max_seq_length]
        else:
          tokens = tokens[-max_seq_length:]

      vocab_words = tokenizer.vocab_words()

      masked_tokens, mask_indexes, masked_labels = create_masked_lm_predictions(
                                                tokens, FLAGS.masked_lm_prob, 
                                                FLAGS.max_predictions_per_seq, vocab_words, rng)

      masked_token_ids = tokenizer.tokens_to_ids(masked_tokens)
      masked_label_ids = tokenizer.tokens_to_ids(masked_label_ids)

      #pad 0's until length is equal to max_seq_length
      while len(masked_token_ids) < max_seq_length: 
        masked_token_ids.append(0)

      input_mask = [1] * len(masked_token_ids)
      segment_id = [0] * len(masked_token_ids)
      mask_weights = [1.0] * len(masked_token_ids)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length
      assert len(mask_indexes) == len(masked_label_ids)

      
      training_instance = TrainingInstance(masked_token_ids, input_mask, segment_id,
                                            mask_indexes, masked_label_ids,  mask_weights)
      
      dataWriter.write(training_instance)



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  tf.logging.info("number of instances: %i", len(instances))

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
