#collect test results

from optparse import OptionParser
import numpy as np
from os import listdir
dir_ = 'results/MARCC_results/'

bold_best_n = 3

parser = OptionParser()
parser.add_option("-F", "--filter", action="store", default=.1, type=float, dest="filter_min")
(options, args) = parser.parse_args()
dir_ = args[0]

results_files = sorted([ fname for fname in listdir(dir_) if fname.split('_')[-1] == 'results.txt' ])

output = []
for fname in results_files:
	fields = fname.split('_')[0].split(':'); 
	model_name = fname.split('_')[0]
	print(fname)
	with open(dir_+fname,'r') as f: 
		results = [ [ ll.split() for ll in l.strip().split('\t\t') ] for l in f.readlines() ]
	if len(results[0]) == 11:
		results = [ [ l[0][0][0] ] + [ l[0][0][1:], l[0][0][1] ] + l[1:] for l in results ]
	#index of hits10: 8, 11
	#try: 
	float_results = [ [ int(l[0][0]), (float(l[2][1])+float(l[3][1]))/2., (float(l[4][1])+float(l[5][1]))/2.,  (float(l[6][1]) + float(l[9][1]))/2.,  (float(l[7][1]) + float(l[10][1]))/2., (float(l[8][1]) + float(l[11][1]))/2. ] for l in results ]
	max_line = max( list(enumerate(float_results)), key=lambda t: t[1][-1] )
	print(max_line)
	out_ = [model_name] + max_line[1] 
	output.append(out_)

#cols = len(output[0]); 
higher_is_better = [ False, True, True, True, True ]


import re
output_ = [ line for line in output if not re.search('TEST', line[0]) ]

output_fb = [ line for line in output_ if re.search('freebase', line[0]) ]
cols_fb = [  np.array([ line_[j] for line_ in output_fb ]) for j in range(2,7) ]
sorted_cols_fb = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_fb) ]

out_str = '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_fb):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= options.filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_fb[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} \n\n'

output_wn = [ line for line in output_ if re.search('wordnet', line[0]) ]
cols_wn = [  np.array([ line_[j] for line_ in output_wn ]) for j in range(2,7) ]
sorted_cols_wn = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_wn) ]; print(sorted_cols_wn)
out_str += '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_wn):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= options.filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_wn[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} \n\n'

output_test = [ line for line in output if re.search('TEST', line[0]) ]

output_test_fb = [ line for line in output_test if re.search('freebase', line[0]) ]
cols_test_fb = [  np.array([ line_[j] for line_ in output_test_fb ]) for j in range(2,7) ]
sorted_cols_test_fb = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_test_fb) ]

out_str += '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_test_fb):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= options.filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_test_fb[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} \n\n'

output_test_wn = [ line for line in output_test if re.search('wordnet', line[0]) ]
cols_test_wn = [  np.array([ line_[j] for line_ in output_test_wn ]) for j in range(2,7) ]
sorted_cols_test_wn = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_test_wn) ]; print(sorted_cols_test_wn)
out_str += '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_test_wn):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= options.filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_test_wn[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} \n\n'

svo_filter_min = 0.0
output_svo = [ line for line in output_ if re.search('svo', line[0]) ]
cols_svo = [  np.array([ line_[j] for line_ in output_svo ]) for j in range(2,7) ]
sorted_cols_svo = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_svo) ]; print(sorted_cols_svo)
out_str += '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_svo):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= svo_filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_svo[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} \n\n'

#ThirdOrder models
to_filter_min = 0.0
output_to = [ line for line in output_ if re.search('ThirdOrder', line[0]) ]
cols_to = [  np.array([ line_[j] for line_ in output_to ]) for j in range(2,7) ]
sorted_cols_to = [ np.argsort(col)[::-1] if higher_is_better[j] else np.argsort(col) for j, col in enumerate(cols_to) ]; print(sorted_cols_to)
out_str += '\\begin{tabular}{lllllll} \\textbf{Model} & \\textbf{Epochs} & \\textbf{MR} & \\textbf{MRR} & \\textbf{Hits@1} & \\textbf{Hits@3} & \\textbf{Hits@10} \\\\ \n '
for i, (fname, epoch, MR, MRR, H1, H3, H10) in enumerate(output_to):
	#for i, line in enumerate(output):
	obj_list = (MR, MRR, H1, H3, H10)
	if H10 >= to_filter_min:
		line_str = ' %s & %i ' % (fname, epoch)
		#cols = range(2,7)
		for j, obj in enumerate(obj_list):
			col = sorted_cols_to[j]
			if i in col[:bold_best_n]:
				result_to_record = ' \\textbf{%.3f} ' % obj
			else:
				result_to_record = ' %.3f ' % obj
			line_str += '& '+result_to_record+' '
		out_str += line_str
		#out_str += ' %s & %i & %.3f & %.3f & %.3f & %.3f & %.3f ' % (fname, epoch, MR, MRR, H1, H3, H10)
	#out_str += ' & '.join( [ str(obj_) for obj_ in line ])
		if i < len(output)-1: out_str += ' \\\\ \n'

out_str += ' \end{tabular} '


with open('results_table.tex', 'w') as outfile: outfile.write(out_str) 
