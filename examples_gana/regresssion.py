import subprocess
import os
import io
import re
import matplotlib.pyplot as plt
import json
import logging

if not os.path.exists('LOG'):
    os.makedirs('LOG')
logging.basicConfig(filename='LOG/logfile.log', level=logging.DEBUG)

#Epoch: 01, Loss: 0.7008, Val: 0.4035, Test: 0.3569
p=re.compile(r'Epoch: (\d+), Loss: (\d+\.\d*), Val: (\d+\.\d*), Test: (\d+\.\d*)$')


if os.path.exists('results/summary.json'):
	with open('results/summary.json', 'r') as f:
		summary = json.load(f)
else:
	summary = {}

def run_regression(file_name, k, epochs,hidden_channels=16):
	dict_key = '_'.join([file_name, str(k), str(epochs), str(hidden_channels)])
	if dict_key in summary.keys():
		logging.info(f"skipping run {file_name,k,epochs, hidden_channels} as summary exists")
		return
	arg0 = f'{file_name}.py'
	arg1 = f'-k={k}'
	arg2 = f'-epochs={epochs}'
	arg3 = f'-hidden_channels={hidden_channels}'
	logging.info(f"running {arg0} with {arg1} {arg2} {arg3}")
	print(f"running {arg0} with {arg1} {arg2} {arg3}")
	proc = subprocess.Popen(['python',arg0, arg1, arg2, arg3],stdout=subprocess.PIPE)
	epoch_list = []
	loss_list = []
	val_list = []
	test_list = []
	for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
		if p.match(line):
			epoch, loss, val, test = p.match(line).groups()
			epoch_list.append(int(epoch))
			loss_list.append(float(loss))
			val_list.append(float(val))
			test_list.append(float(test))
	summary[dict_key] ={'epoch': epoch_list,
							'loss': loss_list, 'val':val_list, 'test':test_list}
def plot_results(xlimit, ylimit, title, dict_key, fpath):
	fig,ax = plt.subplots()
	for key, value in summary.items():
		ax.plot(value[dict_key], label=key)
	ax.set_ylim(0,xlimit)
	ax.set_xlim(0,ylimit)
	ax.set_ylabel(title, fontsize=14,fontweight= 'bold')
	ax.set_xlabel('# Epochs', fontsize=14,fontweight= 'bold')
	ax.set_title(f'{title} with epochs', fontsize=14, fontweight='bold')
	ax.legend()
	if os.path.exists(fpath):
		os.remove(fpath)
	fig.savefig(fpath)

if __name__ == '__main__':

	file_name = 'gana_gat'
	# file_name = 'gana_gat' #high error
	# file_name = 'gana_sage' #epochs 101
	# file_name = 'gana_rgcn' #epochs 41
	# file_name = 'gana_rgat' #epochs 41
	epochs = 101
	# run_regression(file_name, 2,epochs)
	# run_regression(file_name, 4,epochs)
	run_regression(file_name, 6,epochs)
	# run_regression(file_name, 8,epochs)
	# # run_regression(file_name, 10,epochs)
	# # run_regression(file_name, 12,epochs)
	# # run_regression(file_name, 14,epochs)
	# # run_regression(file_name, 16,epochs)
	# run_regression(file_name, 4,epochs,8)
	# run_regression(file_name, 4,epochs,24)
	# run_regression(file_name, 4,epochs,32)

	if summary:
		summary_path = f'results/{file_name}_summary.json'
		if os.path.exists(summary_path):
			os.remove(summary_path)
		with open(summary_path, 'w') as f:
			json.dump(summary,f, indent=4)
	# print(summary)

	plot_results(1,epochs-1, 'test accuracy', 'test', f'results/{file_name}_test_comparison.png')
	plot_results(1,epochs-1, 'train accuracy', 'loss', f'results/{file_name}_train_loss_comparison.png')
	plot_results(1,epochs-1, 'validation accuracy', 'val', f'results/{file_name}_val_comparison.png')
