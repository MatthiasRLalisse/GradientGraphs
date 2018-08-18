import tensorflow as tf
import numpy as np
import os, re, sys
from .utils import readTripletsData, defaultValueDict, defaultFalseDict, permuteList
eps = .25	#weight matrices constrained to have l2 norm |W| <= lambda_ - eps

data_presets = defaultFalseDict({'freebase':'./data/FB15K', 'wordnet':'./data/WN18'})	
		#collection of standard datasets packaged here
		#can be passed to KBETaskSetting as 'dataName' arguments

class KBEModel(object):
	def __init__(self, 
		entity_dim=50, 
		relation_dim=None,
		h_dim=None,
		n_entity=None,
		n_relation=None,
		lambda_=None,
		lrate=.001,
		name='foo',
		model_dir='model', epoch_num=None ):	#model name allows restoring previous models
		if not relation_dim: relation_dim = entity_dim
		if not h_dim: h_dim = entity_dim*2 + relation_dim
		self.name = name; self.model_dir = model_dir
		self.entity_dim = entity_dim; self.relation_dim = relation_dim; self.h_dim=h_dim
		self.n_entity = n_entity; self.n_relation = n_relation
		self.lrate = lrate
		self.lambda_=lambda_
		self.build_embeddings()
				
		self.x = self.build_x()
		self.W, self.b = self.build_params()
		self.mu_h = self.build_mu_h()
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
			self.e_embeddings = tf.get_variable("entityEmbeddings", shape=[self.n_entity, self.entity_dim],
        	   		initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			self.r_embeddings = tf.get_variable("relationEmbeddings", shape=[self.n_relation, self.relation_dim],
        	   		initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		
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

	def build_params(self):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			W_ = tf.get_variable("W_", shape=[self.h_dim, self.h_dim], 
				initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
			if self.lambda_:
				W = tf.clip_by_norm((tf.transpose(W_) + W_)/2., self.lambda_-eps)
			else:
				W = tf.clip_by_norm((tf.transpose(W_) + W_)/2., 2.5)
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
			mu_h = tf.tensordot(self.m_x,self.Sigma,axes=[2,0])/2.
			return mu_h

	def build_H(self):
		if self.lambda_:
			H = -1./2.*(tf.reduce_sum(tf.multiply(self.mu_h,tf.tensordot(self.mu_h,\
				-self.W,axes=[2,0])),axis=2) - tf.tensordot(self.mu_h,self.b,axes=[2,0])\
				+ self.lambda_*tf.reduce_sum(tf.square(self.mu_h -
				self.x),axis=2))
		else:	#if lambda_ is not specified, then self.mu_h == self.x (i.e. lambda_ == inf)
			H = -1./2.*(tf.reduce_sum(tf.multiply(self.mu_h,tf.tensordot(self.mu_h,\
				-self.W,axes=[2,0])),axis=2) - tf.tensordot(self.mu_h,self.b,axes=[2,0]))
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
		return tf.reduce_sum(-tf.nn.log_softmax(self.scores,axis=1)[:,0], axis=0)

	def build_trainer(self):
		self.optimizer = tf.train.AdamOptimizer(self.lrate)
		return self.optimizer.minimize(self.loss)
	
	def restore(self, epoch_num):
		self.saver = tf.train.Saver(max_to_keep=10)
		re_chk = re.compile(re.escape(self.name)+'-epoch_[0-9]+\.ckpt')
		checkpoints = [ file_ for file_ in os.listdir(self.model_dir) if re_chk.match(file_) ]
		epoch_nums = [ int(line_.split('_',1)[1].split('.',1)[0]) for line_ in checkpoints ]
		if len(epoch_nums) > 0:
			if epoch_num:
				self.epoch = epoch_num if epoch_num in epoch_nums else min(epoch_nums)
			else:
				self.epoch = max(epoch_nums)
			print('restoring from epoch {0} model'.format(self.epoch) + '\t'+self.model_dir +\
							'/%s-epoch_%i.ckpt' % (self.name, self.epoch))
			self.saver.restore(self.sess, self.model_dir + \
					('/%s-epoch_%i.ckpt' % (self.name, self.epoch)))
		else: self.epoch = 0

		
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
		if typed: self.n_types = max(self.data['type2idx'].values())

	def get_data(self, dataDirectory, typed):
		return readTripletsData(dataDirectory, typed)
	
	def trainLoop(self, model, e1choice, rchoice, e2choice, e1choice_neg, e2choice_neg, sess=None, e1types=None):
		if not sess: sess = model.sess
		batch_size = len(rchoice)
		e1_choice_ = [ [ e1choice[j] for i in range(self.negsamples+1) ] for j in range(batch_size) ]
		e1_choice_neg = [ [ e1choice[j] ] + e1choice_neg[j] for j in range(batch_size) ]
		r_choice_ = rchoice 
		e2_choice_ = [ [ e2choice[j] for i in range(self.negsamples+1) ] for j in range(batch_size) ]
		e2_choice_neg = [ [ e2choice[j] ] + e2choice_neg[j] for j in range(batch_size) ]
		#train left entity
		batch_loss_left, null = sess.run([model.loss, model.train], {
					model.e1_choice: e1_choice_neg, 
					model.r_choice: r_choice_,
					model.e2_choice: e2_choice_})
		batch_loss_left = np.sum(batch_loss_left)
		#train right entity
		if e1types:
			batch_loss_right, null = sess.run([model.loss, model.train], {
					model.e1_choice: e1_choice_, 
					model.r_choice: r_choice_, 
					model.e2_choice: e2_choice_neg,
					model.e1type_choice: e1types,
					model.train_classes: True }) 
		else:
			batch_loss_right, null = sess.run([model.loss, model.train], {
					model.e1_choice: e1_choice_, 
					model.r_choice: r_choice_, 
					model.e2_choice: e2_choice_neg })
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
			batch_loss = self.trainLoop(model, e1choice, rchoice, e2choice, \
							e1choice_neg, e2choice_neg, sess=sess, e1types=e1types)
			epoch_error += batch_loss
			if interactive: 
				sys.stdout.flush(); 
				sys.stdout.write(('\rtraining epoch %i \tbatch %i of %i \tbatch loss = %f\t\t'\
								% (epoch, i+1, batches_, batch_loss))+'\r')
			#save trained model
		model_path = model.model_dir + '/' + model.name + '-epoch_%i.ckpt' % (epoch,)
		model.saver.save(sess, model_path)
		model.epoch += 1

	def rankEntities(self, model, entity_1s,relations_,entity_2s, sess=None, direction='r'):
		if not sess: sess = model.sess
		true_triplets = (entity_1s,relations_,entity_2s)
		candidates_ = []
		entities_ = []; relations__ = []
		for j in range(len(entity_1s)):
			entity_1, relation_, entity_2 = entity_1s[j], relations_[j], entity_2s[j]
			if direction == 'r':	
				candidates = [ [entity_2] + [ e_ for e_ in self.data['candidates_r'][relation_] \
					if e_ != entity_2 and not(self.data['filter'][(entity_1,relation_,e_)]) ] ]
				entities_ += [[ entity_1 for i in c ] for c in candidates ]
			else:
				candidates = [[entity_1] + [ e_ for e_ in self.data['candidates_l'][relation_] \
					if e_ != entity_1 and not(self.data['filter'][(e_,relation_,entity_2)]) ] ]
				entities_ += [[ entity_2 for i in c ] for c in candidates ]
			candidates_ += candidates
		if direction=='r':
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

	def eval(self, model, sess=None, test_set=False, interactive=False):
		if not sess: sess = model.sess
		print('testing...\n') 
		if test_set: 
			eval_data = self.data['test']
		else:
			eval_data = self.data['valid']
		e1s_test, rs_test, e2s_test = [ d[:1000] for d in eval_data[:3] ]
		test_batch_size = 1
		perm_ = np.random.permutation(len(e2s_test))
		e1s_test_p, rs_test_p, e2s_test_p = [ permuteList(l, perm_) for l in [e1s_test,rs_test,e2s_test] ]
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
					'MRR left = %.5f, Hits@10 left = %.5f, MRR right = %.5f, '\
					'Hits@10 right = %.5f\t\t\r' % \
					(k+1, test_batches_, mean_rank_e1, mean_rank_e2, \
					MRR_left, hits_10l/n_examples, MRR_right, \
					hits_10r/n_examples))
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
		return results
		
