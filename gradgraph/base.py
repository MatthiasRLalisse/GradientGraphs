import tensorflow as tf
import numpy as np
import os, re, sys
import gradgraph as gg

path = gg.__path__[0]
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

from .utils import readTripletsData, defaultValueDict, defaultFalseDict, permuteList
eps = .25	#weight matrices constrained to have l2 norm |W| <= lambda_ - eps

trained_models_path = os.path.join(path, 'trained_models')
data_presets = defaultFalseDict({'freebase':path+'/data/FB15K', 'wordnet':path+'/data/WN18', \
					'svo':path+'/data/SVO', 'fb237':path+'/data/FB15K-237',
					'wn18rr':path+'/data/WN18RR', \
					'yago3-10': path+'/data/YAGO3-10'})	
		#collection of standard datasets packaged with gradgraph
		#can be passed to KBETaskSetting as 'dataName' arguments to automate dataloading

class KBEModel(object):
	def __init__(self, 
		entity_dim=50, 
		relation_dim=None,
		task=None,
		h_dim=None,
		n_entity=None,
		n_relation=None,
		n_types=0,
		lambda_=None,
		gamma=0.,
		train_dropout=0.,
		lrate=.001,
		name=None,
		model_dir=None, 
		epoch_num=None,
		dataName='DataUnKnown',
		trip=False ):	#model name allows restoring previous models
		#self.result_fields = ['MR_left', 'MR_right', 'MRR_left', 'MRR_right', 'Hits1_left', \
		#			'Hits3_left', 'Hits10_left', 'Hits1_right', 'Hits3_right',\
		#			'Hits10_right', 'Hits10', 'Hits3', 'Hits1', 'MRR', 'MR']
		if not relation_dim: relation_dim = entity_dim
		if not h_dim: h_dim = entity_dim*2 + relation_dim
		self.name = name
		self.model_dir = trained_models_path if not model_dir else model_dir
		try: 
			with open(os.path.join(self.model_dir, self.name+'-hyperparams.txt'),'r') as f:
				lines = { l.split()[0]: l.split()[1] for l in f.readlines() }
			print( 'restoring hyperparams from save file')
			self.dataName = lines['dataName']
			self.entity_dim = int(lines['entity_dim'])
			self.relation_dim = int(lines['relation_dim'])
			self.h_dim = int(lines['h_dim'])
			self.lambda_ = None if lines['lambda']=='None' \
							else float(lines['lambda'])
			self.n_entity = int(lines['n_entity'])
			self.n_relation = int(lines['n_relation'])
			self.gamma = float(lines['gamma'])
			if self.gamma: self.n_types = int(lines['n_types'])
			self.train_dropout = float(lines['train_dropout'])\
						if 'train_dropout' in lines else 0.0
		except FileNotFoundError:
			assert entity_dim and relation_dim and h_dim, \
					'must pass name of a saved model with saved hyperparameters, '\
					'or supply values for entity_dim, relation_dim, and h_dim hyperparams'
			self.entity_dim = entity_dim; self.relation_dim = relation_dim
			self.h_dim=h_dim; self.lambda_ = lambda_; self.gamma = gamma	
			self.train_dropout = train_dropout
			if task:
				self.dataName = task.dataName
				self.n_entity = task.n_entity; self.n_relation = task.n_relation
				if task.typed: self.n_types = task.n_types
			else: 
				assert n_entity and n_relation, \
					'ERROR must pass kwarg task or kwargs n_entity and n_relation'
				self.n_entity = n_entity; self.n_relation = n_relation
				self.dataName = dataName
				if self.gamma:
					assert n_types, 'ERROR must pass n_types if gamma is not 0'
		self.hyperparams = { 'dataName': self.dataName, 'entity_dim': self.entity_dim, \
						'relation_dim': self.relation_dim, 'h_dim': self.h_dim, \
						'lambda': self.lambda_, 'n_entity':self.n_entity,\
						'n_relation': self.n_relation, 'gamma': self.gamma,
						'train_dropout': self.train_dropout }	
		self.lrate = lrate; self.gamma=gamma
		self.build_embeddings()
		self.x = self.build_x()
		self.W, self.b = self.build_params()
		self.mu_h = self.build_mu_h()
		try: 
			self.mu_h_1, self.mu_h_2 = self.mu_entities()
		except NotImplementedError: pass
		if gamma: 
			assert self.n_types, 'specify the number of types (perhaps by inputting a typed task)'
			self.hyperparams['n_types'] = self.n_types
			try:
				self.class_probs = self.build_classifier(trip)
			except NotImplementedError: 
				print('token embeddings not defined for this model--'\
							'cannot build classification loss')
				raise NotImplementedError()
		self.H = self.build_H()
		self.scores = self.build_scores()
		self.loss = self.build_loss()
		self.train = self.build_trainer()
		self.sess = tf.Session()
		init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(init)
		self.restore(epoch_num)

	def build_embeddings(self):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.e_embeddings = tf.get_variable("entityEmbeddings", 
				shape=[self.n_entity, self.entity_dim],
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.r_embeddings = tf.get_variable("relationEmbeddings", 
				shape=[self.n_relation, self.relation_dim], initializer=
				tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.e1_choice = tf.placeholder(tf.int32, shape=[None, None])
		self.r_choice = tf.placeholder(tf.int32, shape=[None])
		self.e2_choice = tf.placeholder(tf.int32, shape=[None, None])

		e1_embed = tf.nn.embedding_lookup(self.e_embeddings, self.e1_choice)
		r_embed = tf.nn.embedding_lookup(self.r_embeddings, self.r_choice)
		e2_embed = tf.nn.embedding_lookup(self.e_embeddings, self.e2_choice)

		self.e1 = tf.nn.l2_normalize(e1_embed, axis=2)
		self.r = tf.expand_dims(tf.nn.l2_normalize(r_embed, axis=1),1)
		self.e2 = tf.nn.l2_normalize(e2_embed, axis=2)
	
	def build_x(self):	#define this in model subclass
		raise NotImplementedError()

	def mu_entities(self):
		raise NotImplementedError()

	def build_params(self):	
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.drop_prob = tf.placeholder_with_default(0.0, shape=())
			self.W_ = tf.get_variable("W_", shape=[self.h_dim, self.h_dim], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.W_drop = tf.nn.dropout(self.W_, keep_prob=1.-self.drop_prob)
			if self.lambda_:
				W = tf.clip_by_norm((tf.transpose(self.W_drop) + self.W_drop)/2., self.lambda_-eps)
			else:
				W = tf.clip_by_norm((tf.transpose(self.W_drop) + self.W_drop)/2., 2.5)
			b = tf.get_variable("b_", shape=[self.h_dim], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		return W, b

	
	def build_mu_h(self):
		if not self.lambda_:
			return self.x
		else:
			b_ = tf.expand_dims(tf.expand_dims(self.b,0),0)
			self.lambdaI = self.lambda_*tf.eye(self.h_dim,dtype=tf.float32)
			self.Sigma_inv = self.lambdaI - self.W
			self.Sigma = tf.matrix_inverse(self.Sigma_inv)
			self.m_x = 2*self.lambda_*self.x + b_
			return tf.tensordot(self.m_x,self.Sigma,axes=[2,0])/2.

	def build_H(self):
		if self.lambda_:
			H = -1./2.*(tf.einsum('bni,ij,bnj->bn',self.mu_h,-self.W,self.mu_h) - \
				tf.tensordot(self.mu_h,self.b,axes=[2,0]) + self.lambda_*\
				tf.reduce_sum(tf.square(self.mu_h - self.x), axis=2))
		else:	
			H = -1./2.*(tf.einsum('bni,ij,bnj->bn',self.mu_h,-self.W,self.mu_h) - \
				tf.tensordot(self.mu_h,self.b,axes=[2,0]))
		return H

	def build_scores(self):	#by default, scores are Harmony values
		scores = self.H
		pos_scores = tf.expand_dims(scores[:,0],1)
		neg_scores = scores[:,1:]
		self.softmax_scores = tf.nn.softmax(scores,axis=1)
		self.test_scores = tf.squeeze(self.softmax_scores, axis=0)
		self.true_posterior = self.softmax_scores[:,0]
		return scores

	def build_loss(self):	#default loss is log-softmax of the true triplet
		self.rank_loss = tf.reduce_sum(-tf.nn.log_softmax(self.scores,axis=1)[:,0], axis=0)
		if self.gamma:
			self.true_class_indices = tf.stack([tf.range(tf.shape(self.e1type_choice)[0]), \
								self.e1type_choice], axis=1)
			self.class_loss = tf.reduce_sum(-tf.gather_nd(self.log_class_probs, \
								self.true_class_indices), axis=0)
			return (1-self.gamma)*self.rank_loss + tf.cond(self.train_classes, \
					true_fn=lambda: self.gamma*self.class_loss, false_fn=lambda: 0.)
		else:
			return self.rank_loss

	def build_trainer(self):
		self.optimizer = tf.train.AdamOptimizer(self.lrate)
		return self.optimizer.minimize(self.loss)

	@property
	def params(self):
		W, b, e_embed, r_embed = self.sess.run([self.W, self.b, \
						self.e_embeddings, self.r_embeddings])
		params = { 'weights': W, 'bias': b, 'entity_embeddings': e_embed, \
							'relation_embeddings': r_embed }
		if self.gamma: 
			params['W_classifier'] = self.W_class
			params['b_classifier'] = self.b_class
		return { 'weights': W, 'bias': b, 'entity_embeddings': e_embed, \
							'relation_embeddings': r_embed }

	def build_classifier(self, trip=True):
		self.e1type_choice = tf.placeholder_with_default(tf.constant([0],dtype=tf.int32), shape=[None])
		self.train_classes = tf.placeholder_with_default(False, shape=())
		self.W_class = tf.get_variable('W_class', shape=[self.n_types, self.h_dim], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.b_class = tf.get_variable("b_class", shape=[self.n_types], \
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		#true_mu_h_1 = self.mu_h_1[:,0,:]	#true entity is the first in negative sample dim
		if trip: true_mu_h = self.mu_h[:,0,:]
		else: true_mu_h = self.mu_h_1[:,0,:]
		class_layer = tf.nn.tanh(tf.einsum('ij,bj->bi', self.W_class, true_mu_h) + 
						tf.expand_dims(self.b_class, 0)) #tf.nn.sigmoid...
		class_probs = tf.nn.softmax(class_layer, axis=1)
		self.log_class_probs = tf.nn.log_softmax(class_layer, axis=1)
		#true_class_posterior = tf.gather_nd(class_probs, tf.stack([tf.range(\
		#			tf.shape(self.e1type_choice)[0]), self.e1type_choice], axis=1))
		#self.gamma_ = tf.cond(self.train_classes, true_fn=lambda:\
		#			tf.constant(self.gamma), false_fn=lambda:0.)
		return class_probs
	
	def restore(self, epoch_num):
		self.saver = tf.train.Saver(max_to_keep=10)
		re_chk = re.compile(re.escape(self.name)+'-epoch_[0-9]+\.ckpt')
		checkpoints = [ file_ for file_ in os.listdir(self.model_dir) if re_chk.match(file_) ]
		epoch_nums = [ int(line_.split('_',1)[1].split('.',1)[0]) for line_ in checkpoints ]
		if len(epoch_nums) > 0:
			if epoch_num:
				if epoch_num in epoch_nums:
					self.epoch = epoch_num 
				else:
					self.epoch = max(epoch_nums)
					print('could not find model matching epoch_num %i\nrestoring'\
							' epoch %i instead' % (epoch_num,self.epoch))
			else:
				self.epoch = max(epoch_nums)
			model_id = '%s-epoch_%i.ckpt' % (self.name, self.epoch)
			print('restoring from epoch {0} model'.format(self.epoch) + '\t' \
							+ os.path.join(self.model_dir, model_id))
			self.saver.restore(self.sess, os.path.join(self.model_dir, model_id))
			try: 
				self.results = { (int(line.split('\t')[0].split()[1]), line.split('\t')[0].split()[2]) : \
						{ obj.split()[0]: float(obj.split()[1]) for obj in line.split('\t')[1:] } \
						for line in (open(os.path.join(self.model_dir, self.name+'-results.txt'), 'r')).readlines() }
			except FileNotFoundError:
				self.results = {}
		else: 
			self.epoch = 0; self.results = {}
	
	def save(self, force_overwrite=False):
		#save trained model
		model_id = self.name + '-epoch_%i.ckpt' % (self.epoch,)
		model_path = os.path.join(self.model_dir, model_id)
		if not force_overwrite and any( re.match(re.compile(model_id+'*'), filename) for \
						filename in os.listdir(self.model_dir)):
			overwrite_ = input('data for model %s already exists--overwrite?'\
					' (type \'YES\' to overwrite) ' % (model_id,))
			if overwrite_ != 'YES': 
				print('model not saved'); return
		self.saver.save(self.sess, model_path)
		result_lines = [ 'epoch '+str(epoch) + ' ' + datatype + '\t' + '\t'.join([key + ' '+ \
				str(self.results[(epoch,datatype)][key]) for key in self.results[(epoch, datatype)] ]) \
				for epoch, datatype in sorted(self.results.keys()) ]
		for i in range(len(result_lines)-1): result_lines[i] += '\n'
		with open(os.path.join(self.model_dir, self.name + '-results.txt'), 'w') as f:
			f.writelines(result_lines)
		hyperparams_lines = [ key + ' ' + str(self.hyperparams[key]) + '\n' for key in self.hyperparams ]
		hyperparams_lines[-1] = hyperparams_lines[-1][:-1]
		hpfile = os.path.join(self.model_dir, self.name+'-hyperparams.txt')
		with open(hpfile, 'w') as f:
			f.writelines(hyperparams_lines)
		print('saved at %s' % (model_path,)); return

	def predict(self, e1_, r_, e2_):
		x, mu_h = self.sess.run( [self.x, self.mu_h], { self.e1_choice:[[e1_]], \
						self.r_choice:[r_], self.e2_choice:[[e2_]]})
		return np.squeeze(np.squeeze(x, axis=0),axis=0), np.squeeze(np.squeeze(mu_h, axis=0),axis=0)

	def predict_token(self, e1_, r_, e2_, direction='r'):
		target, type_embed = ('mu_h_2', self.e2) if direction=='r' else ('mu_h_1', self.e1)
		if not hasattr(self, target):
			raise NotImplementedError
		e_, mu_h_ = self.sess.run( [type_embed, getattr(self, target)], {self.e1_choice:[[e1_]], \
							self.r_choice:[r_], self.e2_choice:[[e2_]]})
		e_out = np.squeeze(np.squeeze(e_, axis=0), axis=0)
		mu_h_out = np.squeeze(np.squeeze(mu_h_, axis=0), axis=0)
		return e_out, mu_h_out
		
class KBETaskSetting(object):
	def __init__(self,
		dataName = None,
		dataDirectory='./data/',
		typed=False,
		negsamples=100,
		batch_size=100,
		filtered=True,		#if False, use raw eval setting
		type_constrain=True):	#if True, candidate entities must satisfy relational type constraints
		self.negsamples = negsamples
		self.batch_size = batch_size
		self.typed = typed; self.filtered = filtered; self.type_constrain = type_constrain
		if data_presets[dataName]: 
			dataDirectory = data_presets[dataName]; self.dataName = dataName
		else: self.dataName = dataDirectory.split('/')[-1]
		self.data = self.get_data(dataDirectory, typed)
		self.n_entity = max(self.data['entity2idx'].values())+1
		self.n_relation = max(self.data['relation2idx'].values())+1
		if not self.filtered: self.data['filter'] = defaultFalseDict()
		if not self.type_constrain: 
			constraints = defaultValueDict(); constraints.set_default(list(range(self.n_entity)))
			self.data['candidates_l'] = self.data['candidates_r'] = constraints
		if typed: self.n_types = max(self.data['type2idx'].values())+1

	def get_data(self, dataDirectory, typed):
		return readTripletsData(dataDirectory, typed)
	
	def trainLoop(self, model, e1choice, rchoice, e2choice, e1choice_neg, e2choice_neg, sess=None, e1types=None):
		if not sess: sess = model.sess
		batch_size = len(rchoice)
		e1_choice_ = [ [e1choice[j]]*(self.negsamples+1) for j in range(batch_size) ]
		e1_choice_neg = [ [ e1choice[j] ] + e1choice_neg[j] for j in range(batch_size) ]
		r_choice_ = rchoice 
		e2_choice_ = [ [e2choice[j]]*(self.negsamples+1)  for j in range(batch_size) ]
		e2_choice_neg = [ [ e2choice[j] ] + e2choice_neg[j] for j in range(batch_size) ]
		left_placeholders = {	model.e1_choice: e1_choice_neg, 
					model.r_choice: r_choice_,
					model.e2_choice: e2_choice_,
					model.drop_prob: model.train_dropout }
		right_placeholders =  {	model.e1_choice: e1_choice_, 
					model.r_choice: r_choice_, 
					model.e2_choice: e2_choice_neg,
					model.drop_prob: model.train_dropout }
		if e1types: 
			left_placeholders[model.e1type_choice] = e1types
			left_placeholders[model.train_classes] = True
			#right_placeholders[model.e1type_choice] = e1types
		#train left entity
		batch_loss_left, null = sess.run([model.loss, model.train], left_placeholders )
		batch_loss_left = np.sum(batch_loss_left)
		#train right entity
		batch_loss_right, null = sess.run([model.loss, model.train], right_placeholders)
		batch_loss_right = np.sum(batch_loss_right)
		return batch_loss_left + batch_loss_right

	def trainEpoch(self, model, sess=None, interactive=False):
		if not sess: sess = model.sess
		epoch = model.epoch + 1
		e1s_train, rs_train, e2s_train = self.data['train'][:3]
		if self.typed: e1types_train = self.data['train'][3]
		batches_ = int(len(e1s_train)/self.batch_size)
		perm_ = np.random.permutation(len(e2s_train))
		e1s_train_p, rs_train_p, e2s_train_p = [ permuteList(l, perm_) for l in \
							[e1s_train, rs_train, e2s_train] ]
		if self.typed: e1types_train_p = permuteList(e1types_train, perm_)
		print('epoch {0}'.format(epoch))
		epoch_error = 0
		for i in range(batches_):
			if self.typed:
				e1choice, rchoice, e2choice, e1types = [ l[i*self.batch_size:i*self.batch_size + \
					self.batch_size] for l in [e1s_train_p, rs_train_p, \
					e2s_train_p, e1types_train_p ] ]
			else:
				e1choice, rchoice, e2choice = [ l[i*self.batch_size:i*self.batch_size + \
					self.batch_size] for l in [e1s_train_p, rs_train_p, \
					e2s_train_p ] ]
				e1types = None
			e1choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e1choice)) ]
			e2choice_neg = [ [ np.random.randint(self.n_entity) for n in range(self.negsamples) ] \
									for m in range(len(e2choice)) ]
			#e1choice_neg = list(np.random.randint(0, high=self.n_entity, size=(len(e1choice),self.negsamples)))
			#e2choice_neg = list(np.random.randint(0, high=self.n_entity, size=(len(e2choice),self.negsamples)))
			#print(e1choice_neg);print(e2choice_neg)
			batch_loss = self.trainLoop(model, e1choice, rchoice, e2choice, \
							e1choice_neg, e2choice_neg, sess=sess, e1types=e1types)
			epoch_error += batch_loss
			if interactive: 
				sys.stdout.flush(); 
				sys.stdout.write(('\rtraining epoch %i \tbatch %i of %i \tbatch loss = %f\t\t'\
								% (epoch, i+1, batches_, batch_loss))+'\r')
		model.epoch += 1
		return epoch_error
		

	def rankEntities(self, model, entity_1s,relations_,entity_2s, direction='r', \
					 sess=None, type_constrain=None, filtered=None):
		if not sess: sess = model.sess; 
		if type_constrain or type_constrain==None:
			candDict = self.data['candidates_'+direction]
		else: 
			candDict = defaultValueDict(); 
			candDict.set_default(list(range(model.n_entity)))
		filtered = filtered if filtered != None else self.filtered
		Filter = self.data['filter'] if filtered else defaultFalseDict()
		true_triplets = (entity_1s,relations_,entity_2s)
		candidates_ = []
		entities_ = []; relations__ = []
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			if direction == 'r':	
				candidates = [ [entity_2] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_2 and not(Filter[(entity_1,relation_,e_)]) ] ]
				entities_ += [[entity_1]*len(candidates) ]
			else:
				candidates = [[entity_1] + [ e_ for e_ in candDict[relation_] \
					if e_ != entity_1 and not(Filter[(e_,relation_,entity_2)]) ] ]
				entities_ += [[entity_2]*len(candidates) ]
			candidates_ += candidates
		if direction=='r':
			#neg_dim, batch_dim, x, ravel1, scores =sess.run([model.negdim, model.batchdim, model.x, model.ravel1, model.test_scores], {model.e1_choice: entities_,
								#model.r_choice: relations_,
								#model.e2_choice: candidates_ })
			#print(neg_dim, batch_dim, x.shape, ravel1.shape, scores.shape)
			scores = [sess.run( model.test_scores, {model.e1_choice: entities_,
								model.r_choice: relations_,
								model.e2_choice: candidates_ })]
		else:
			scores = [sess.run( model.test_scores, {model.e1_choice: candidates_,
								model.r_choice: relations_,
								model.e2_choice: entities_ })]
		candidates_perms = [ sorted( range(len(candidates)), key=lambda x:scores[j][x] )[::-1] \
							for j,candidates in enumerate(candidates_) ]
		ranked = [ [ candidates[i] for i in candidates_perms[j] ] for j,candidates in enumerate(candidates_) ]
		return ranked

	def rank(self, model, entity_1, relation_, entity_2, sess=None, direction='r'):
		if not sess: sess = model.sess
		ranked_entities = self.rankEntities(model, entity_1, relation_, entity_2, \
									sess=sess, direction=direction)
		if direction=='r':
			rank = [ ranks_.index(entity_2[j])+1 for j, ranks_ in enumerate(ranked_entities) ]
		else:
			rank = [ ranks_.index(entity_1[j])+1 for j,ranks_ in enumerate(ranked_entities) ]
		return rank

	def eval(self, model, sess=None, test_set=False, interactive=False, num_to_test=0):
		if not sess: sess = model.sess
		datatype = 'test' if test_set else 'valid'
		print('testing...\n') 
		eval_data = self.data[datatype]
		e1s_test, rs_test, e2s_test = eval_data[:3]
		test_batch_size = 1
		perm_ = np.random.permutation(len(e2s_test))
		e1s_test_p, rs_test_p, e2s_test_p = [ permuteList(l, perm_) for l in [e1s_test,rs_test,e2s_test] ]
		if num_to_test: 
			e1s_test_p, rs_test_p, e2s_test_p = [ l[:num_to_test] for l in \
									[e1s_test_p, rs_test_p, e2s_test_p] ]
		test_batches_ = int(np.ceil(len(e1s_test_p)/float(test_batch_size)))
		n_examples = 0; ranks_left = []; ranks_right = []
		hits_1l = 0.; hits_3l = 0.; hits_10l = 0.; hits_1r = 0.; hits_3r = 0.; hits_10r = 0.
		for k in range(test_batches_):
			c = k-1
			e1_, r_, e2_ = [ l[k*test_batch_size:k*test_batch_size + test_batch_size] \
							for l in [e1s_test_p, rs_test_p, e2s_test_p ] ]
			n_examples += len(e1_)
			right_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='r')
			right_rank_arr = np.array(right_rank,dtype=np.int32)
			hits_1r += np.sum(right_rank_arr == 1)
			hits_3r += np.sum(right_rank_arr <= 3)
			hits_10r += np.sum(right_rank_arr <= 10)
			left_rank = self.rank(model, e1_,r_,e2_, sess=sess, direction='l')
			left_rank_arr = np.array(left_rank)
			hits_1l += np.sum(left_rank_arr == 1)
			hits_3l += np.sum(left_rank_arr <= 3)
			hits_10l += np.sum(left_rank_arr <= 10)
			ranks_right += right_rank
			ranks_left += left_rank
			mean_rank_e1 = np.sum(left_rank_arr)/float(len(left_rank))
			mean_rank_e2 = np.sum(right_rank_arr)/float(len(right_rank))
			MRR_left = np.sum([ 1./rank_ for rank_ in ranks_left ])/len(ranks_left)
			MRR_right = np.sum([ 1./rank_ for rank_ in ranks_right ])/len(ranks_right)
			if interactive: 
				sys.stdout.flush()
				sys.stdout.write('\r\tbatch %i of %i: rank(e1) = %i \trank(e2) = %i '\
					'MRR = %.5f, Hits@1 = %.5f, Hits@3 = %.5f, '\
					'Hits@10 = %.5f\t\t\r' % \
					(k+1, test_batches_, mean_rank_e1, mean_rank_e2, \
					(MRR_left + MRR_right)/2., \
					(hits_1l + hits_1r)/(n_examples*2), \
					(hits_3l + hits_3r)/(n_examples*2), \
					(hits_10l + hits_10r)/(n_examples*2)))
		if interactive: print('\n')
		mean_rank_left = np.sum(ranks_left)/float(len(ranks_left))
		mean_rank_right = np.sum(ranks_right)/float(len(ranks_right))
		results = {	'MR_left':mean_rank_left,
				'MR_right':mean_rank_right,
				'MRR_left':MRR_left,
				'MRR_right':MRR_right,
				'Hits1_left':hits_1l/n_examples,
				'Hits3_left':hits_3l/n_examples,
				'Hits10_left':hits_10l/n_examples,
				'Hits1_right':hits_1r/n_examples,
				'Hits3_right':hits_3r/n_examples,
				'Hits10_right':hits_10r/n_examples,
				'Hits10': (hits_10l + hits_10r)/(n_examples*2),
				'Hits3': (hits_3l + hits_3r)/(n_examples*2),
				'Hits1': (hits_1l + hits_1r)/(n_examples*2),
				'MRR': (MRR_left + MRR_right)/2.,
				'MR': (mean_rank_left + mean_rank_right)/2. }
		model.results[(model.epoch, datatype)] = results
		return results
		




