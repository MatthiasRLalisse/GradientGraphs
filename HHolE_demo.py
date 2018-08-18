import GradientGraphs as gg
task = gg.KBETaskSetting('wordnet')
model = gg.models.HTPR.model(task.n_entity, task.n_relation, entity_dim=5, relation_dim=5, lambda_=1.)

task.trainEpoch(model.sess, model, interactive=True)
results = task.test(model.sess, model, interactive=True)

print(results['Hits1'], results['Hits3'], results['Hits10'])






