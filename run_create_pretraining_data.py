import subprocess
import sys
import os

#each input file gets its own process 
def launch_data_prep_process(input_file, output_file, masked_lm_prob,
								max_predictions_per_seq, vocab_file="vocab.txt"):

	subprocess.Popen(['python', 'create_pretraining_data.py',
						"--input_file={}".format(input_file),
						"--output_file={}".format(output_file),
						"--vocab_file={}".format(vocab_file),
						"--masked_lm_prob={}".format(masked_lm_prob),
						"--max_predictions_per_seq={}".format(max_predictions_per_seq)])


#list of strings indicating only the file name
input_files = [
	"uniparc_shuffled_part_aa_aa",
	"uniparc_shuffled_part_aa_ab",
	"uniparc_shuffled_part_aa_ac",
	"uniparc_shuffled_part_aa_ad",

	"uniparc_shuffled_part_ab_aa",
	"uniparc_shuffled_part_ab_ab",
	"uniparc_shuffled_part_ab_ac",
	"uniparc_shuffled_part_ab_ad",
]

"""
	"uniparc_shuffled_part_ac_aa",
	"uniparc_shuffled_part_ac_ab",
	"uniparc_shuffled_part_ac_ac",
	"uniparc_shuffled_part_ac_ad",

	"uniparc_shuffled_part_ad_aa",
	"uniparc_shuffled_part_ad_ab",
	"uniparc_shuffled_part_ad_ac",
	"uniparc_shuffled_part_ad_ad"
"""

intput_path_format = "../uniparc/{}"
output_path_format = "../uniparc/output/{}"



"""
input_files = [
	"test.fasta"
]

intput_path_format = "./{}"
output_path_format = "gs://bioinformaticsdatasets/test/{}"
"""

if __name__ == "__main__":
	if len(sys.argv) < 5:
		print('Arguments should be: Input path, Output path, masked_lm_prob, and max_predictions_per_seq')
		exit()

	input_path = sys.argv[1]
	output_path = sys.argv[2]

	masked_lm_prob = sys.argv[3]
	max_predictions_per_seq = sys.argv[4]

	for f in input_files:
		input_path = os.path.join(input_path, f)
		output_path = os.path.join(output_path, f)

		launch_data_prep_process(input_path, output_path, masked_lm_prob, max_predictions_per_seq)

  
