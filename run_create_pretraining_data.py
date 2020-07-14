import subprocess

#each input file gets its own process 
def launch_data_prep_process(input_file, output_file, vocab_file="vocab.txt"):
	subprocess.Popen(['python', 'create_pretraining_data.py',
						"--input_file={}".format(input_file),
						"--output_file={}".format(output_file),
						"--vocab_file={}".format(vocab_file)])


#list of strings indicating only the file name
input_files = [
	"test.fasta",
	"test2.fasta"
]

#intput_path_format = "../uniparc/%s"
#output_path_format = "../uniparc/output/%s"
intput_path_format = "{}"
output_path_format = "process_output/{}"



if __name__ == "__main__":
	for f in input_files:
		input_path = intput_path_format.format(f)
		output_path = output_path_format.format(f)
		launch_data_prep_process(input_path, output_path)
  
