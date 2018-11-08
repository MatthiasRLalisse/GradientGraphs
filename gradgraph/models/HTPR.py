import tensorflow as tf
import numpy as np
from ..base import KBEModel

class HTPR(KBEModel):
	"""Harmonic Tensor Product Representation class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by the tensor product of e1, r, and e2, with dimensionality
	dim(e1)*dim(r)*dim(e2). 
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=5,
		task=None, 
		lambda_=None,
		lrate=.001,
		model_dir='trained_models',
		dataName='DataUnknown',
		name=None,
		epoch_num=None"""
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=5, 
		relation_dim=None,
		task=None,
		lambda_=None,
		gamma=0.0,
		train_dropout=0.,
		lrate=.001,
		model_dir=None,
		dataName=None,
		name=None,
		epoch_num=None ):
		if not relation_dim: relation_dim = entity_dim
		if not name:
			name = 'HTPR%ieD%irD%sL.%s' % (entity_dim, relation_dim, str(lambda_) \
							if lambda_ else 'inf', dataName)
			if gamma: name += 'G%.3f' % (gamma)
		h_dim = relation_dim * entity_dim**2
		super(HTPR, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					task=task,
					lambda_=lambda_,
					gamma=gamma,
					train_dropout=train_dropout,
					lrate=lrate,
					name=name,
					model_dir=model_dir,
					epoch_num=epoch_num)
		self.mu_h_1, self.mu_h_2 = self.mu_entities()

	def build_x(self):
		e1_, r_, e2_ = [ tf.nn.l2_normalize(v)  for v in [self.e1, self.r, self.e2] ]
		e1_r_e2_tpr = tf.einsum('bni,bnj,bnk->bnijk', e1_, r_, e2_)
		self.batchdim = tf.maximum(tf.cast(tf.shape(self.e1_choice)[0], tf.int32), \
						tf.cast(tf.shape(self.e2_choice)[0], tf.int32))
		self.negdim = tf.maximum(tf.cast(tf.shape(self.e1_choice)[1], tf.int32), \
						tf.cast(tf.shape(self.e2_choice)[1], tf.int32))
		self.ravel1 = tf.reshape(e1_r_e2_tpr, [self.batchdim, self.negdim, \
						self.entity_dim, self.relation_dim*self.entity_dim] )
		ravelled_tpr = tf.reshape(self.ravel1, \
						[self.batchdim, self.negdim, self.h_dim])
		return ravelled_tpr
	
	def mu_entities(self):
		unravelled_tpr = tf.reshape(tf.reshape(self.mu_h, [self.batchdim, self.negdim, self.entity_dim, \
					self.relation_dim*self.entity_dim]), [self.batchdim, self.negdim, \
					self.entity_dim, self.relation_dim, self.entity_dim])
		mu_h_1 = tf.einsum('bnk,bnikj,bnj->bni', self.r, unravelled_tpr, self.e2 )
		mu_h_2 = tf.einsum('bni,bnk,bnikj->bnj', self.e1, self.r, unravelled_tpr )
		return mu_h_1, mu_h_2
	


