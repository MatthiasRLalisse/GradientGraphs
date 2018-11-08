import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10


class HDistMult(KBEModel):
	"""Harmonic DistMult class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by the elementwise product of e1, r, and e2:
		x = e1 * r * e2
	kwargs: n_entity=None,
		n_relation=None,
		entity_dim=50,
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
		entity_dim=50, 
		task=None,
		lambda_=None,
		gamma=0.0,
		train_dropout=0.,
		lrate=.001,
		model_dir=None,
		dataName=None,
		name=None,
		epoch_num=None ):
		if not name:
			name = 'DistMult%iD%sL' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf')
			if gamma: name += 'G%.3f' % (gamma)
			name += '-'+ dataName if dataName else (task.dataName if task else 'DataUnKnown')
		relation_dim = h_dim = entity_dim
		super(HDistMult, self).__init__(entity_dim=entity_dim,
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
					epoch_num=epoch_num )
		self.mu_h_1, self.mu_h_2 = self.mu_entities()

	def build_x(self):
		e1_ = tf.nn.l2_normalize(self.e1 + tf.to_float(tf.equal(\
						self.e1, 0.))*no_zeros, axis=2)
		e2_ = tf.nn.l2_normalize(self.e2 + tf.to_float(tf.equal(\
						self.e2, 0.))*no_zeros, axis=2)
		r_ = tf.nn.l2_normalize(self.r + tf.cast(tf.equal(self.r, 0.), \
						dtype=tf.float32)*no_zeros, axis=2)
		#x = tf.multiply(e1_, tf.multiply(r_,e2_))
		x = tf.multiply(self.e1,tf.multiply(self.r, self.e2))
		return x

	def mu_entities(self):
		e1_ = self.e1 + tf.cast(tf.equal(self.e1, 0.), \
						dtype=tf.float32)*no_zeros	#eliminates zero values from division
		r_ = self.r + tf.cast(tf.equal(self.r, 0.), \
						dtype=tf.float32)*no_zeros
		e2_ = self.e2 + tf.cast(tf.equal(self.e2, 0.),\
						dtype=tf.float32)*no_zeros
		mu_h_1 = tf.multiply(tf.multiply(self.mu_h, 1./r_), 1./e2_)
		mu_h_2 = tf.multiply(tf.multiply(self.mu_h, 1./r_), 1./e1_)
		return mu_h_1, mu_h_2



