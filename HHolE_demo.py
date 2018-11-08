import gradgraph as gg

#The KBETaskSetting class implements routines for loading in Knowledge Graph data and training models
task = gg.KBETaskSetting('freebase', negsamples=10, batch_size=200)

#Initialize a GG model. Pre-packaged models are HHolE, HDistMult, and HTPR. 
model = gg.models.HHolE(task.n_entity, task.n_relation, entity_dim=50, lambda_=1., dataName='freebase')

#Train model for one epoch and test. If interactive=True, print incremental progress to stdout
task.trainEpoch(model, interactive=True)

results_valid = task.eval(model, interactive=True) 	#evaluate on validation set
results_test = task.eval(model, test_set=True, interactive=True) 	#evaluate on test set
print('Validation set')
for eval_ in ['Hits1', 'Hits3', 'Hits10' ]:
    print('\t'+results_valid[eval_])

print('Test set')
for eval_ in ['Hits1', 'Hits3', 'Hits10' ]:
    print('\t'+results_test[eval_])





