import subprocess

#each input file gets its own process 
def launch_data_prep_process(input_file, output_file, vocab_file="vocab.txt"):
	subprocess.Popen(['python', 'create_pretraining_data.py',
						"--input_file={}".format(input_file),
						"--output_file={}".format(output_file),
						"--vocab_file={}".format(vocab_file)])


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

	"uniparc_shuffled_part_ac_aa",
	"uniparc_shuffled_part_ac_ab",
	"uniparc_shuffled_part_ac_ac",
	"uniparc_shuffled_part_ac_ad",

	"uniparc_shuffled_part_ad_aa",
	"uniparc_shuffled_part_ad_ab",
	"uniparc_shuffled_part_ad_ac",
	"uniparc_shuffled_part_ad_ad"
]

intput_path_format = "../uniparc/{}"
output_path_format = "../uniparc/output/{}"



if __name__ == "__main__":
	for f in input_files:
		input_path = intput_path_format.format(f)
		output_path = output_path_format.format(f)
		launch_data_prep_process(input_path, output_path)
  
