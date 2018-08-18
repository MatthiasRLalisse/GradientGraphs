import gradgraph as gg
task = gg.KBETaskSetting('freebase', negsamples=250, batch_size=200 )
model = gg.models.HHolE(task.n_entity, task.n_relation, entity_dim=5, lambda_=1., dataName='freebase')

task.trainEpoch(model.sess, model, interactive=True)
results = task.eval(model.sess, model, interactive=True)

print(results['Hits1'], results['Hits3'], results['Hits10'])






