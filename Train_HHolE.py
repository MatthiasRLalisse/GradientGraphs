import gradgraph as gg
import numpy as np
import tensorflow as tf
MAXEPOCH = 500
TEST_EVERY = 1
DIM = 512; BATCHSIZE = 512; NEGSAMPLES = 500

np.random.seed(666)

task = gg.KBETaskSetting('freebase', negsamples=NEGSAMPLES, batch_size=BATCHSIZE, type_constrain=True, filtered=True )
model = gg.models.HHolE(task.n_entity, task.n_relation, entity_dim=DIM, lambda_=1., dataName='freebase')

epochwise_accuracy = []
for e in range(1, MAXEPOCH+1):	
	task.trainEpoch(model, interactive=False)	#train the model for one epoch
	if e % TEST_EVERY == 0: 
		results_valid = task.eval(model, interactive=False) 	#evaluate on validation set
		newline = 'epoch ' + str(model.epoch) + '\t' + \
					'\t'.join( [ key + ' ' + str(results_valid[key]) \
					for key in [ 'MR', 'MRR', 'Hits1', 'Hits3', 'Hits10' ] ]) + '\n'
		acc = results_valid['Hits10']; epochwise_accuracy.append(acc)
		keep_training = ( e < 5 or acc >= np.mean(epochwise_accuracy[-5:]) )	#train at least 5 epochs, and until performance
											#is lower than the moving average
		with open('results/' + model.name + '-results.txt', 'a') as f:
			f.write(newline)
		if not keep_training: break

best_epoch = epochwise_accuracy.index(max(epochwise_accuracy)) + 1
tf.reset_default_graph()
test_model = gg.models.HHolE(task.n_entity, task.n_relation, entity_dim=DIM, \
				lambda_=1., dataName='freebase', epoch_num=best_epoch)
results_test = task.eval(test_model, test_set=True, interactive=False) 	#evaluate on test set

print(results_valid['Hits1'], results_valid['Hits3'], results_valid['Hits10'])

newline = 'TEST\tepoch ' + str(test_model.epoch) + '\t' + '\t'.join( [ key + ' ' + str(results_valid[key]) \
						for key in [ 'MR', 'MRR', 'Hits1', 'Hits3', 'Hits10' ] ])
with open('results/' + test_model.name + '-results.txt', 'a') as f:
	f.write(newline)






