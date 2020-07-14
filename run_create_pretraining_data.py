import subprocess

#each input file gets its own process 
def launch_data_prep_process(input_file, output_file, vocab_file="vocab.txt"):
	subprocess.Popen(['python', 'create_pretraining_data.py',
						"--input_file=\"%s\"".format(input_file),
						"--output_file=\"%s\"".format(output_file),
						"--vocab_file=\"%s\"".format(vocab_file))


#list of strings indicating only the file name
input_files = [
	
]

intput_path_format = "../uniparc/%s"
output_path_format = "../uniparc/output/%s"

if __name__ == "__main__":
	for f in input_files:
		input_path = intput_path_format.format(f)
		output_path = output_path_format.format(f)
		launch_data_prep_process(input_path, output_path)
  
