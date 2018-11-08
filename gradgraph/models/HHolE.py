import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10


class HHolE(KBEModel):
	"""Harmonic Holographic Embedding class. Passes all arguments to the superclass KBEModel. 
	Compositional layer embeddings are given by:
		r * ifft( conj(fft(e1)) * fft(e2))) 	(* is elementwise multiplication)
	which is a computationally efficient formula for the circular convolution of vectors e1, e2.
	
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
		lrate=.001,
		gamma=0.,
		train_dropout=0.,
		model_dir=None,
		dataName=None,
		name=None,
		epoch_num=None):
		if not name: 
			name = 'HHolE%iD%sL' % (entity_dim, str(lambda_) \
						if lambda_ else 'inf')	
							#sets naming convention for this model
			if gamma: name += 'G%.4f' % (gamma)
			name += '-'+ dataName if dataName else (task.dataName if task else 'DataUnKnown')
		relation_dim = h_dim = entity_dim
		super(HHolE, self).__init__(entity_dim=entity_dim,
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
					epoch_num=epoch_num,
					dataName=dataName)
	def build_x(self):
		e1_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e1, axis=2),0.0))
		e2_fft_ = tf.fft(tf.complex(tf.nn.l2_normalize(self.e2, axis=2),0.0))
		self.e1_fft = e1_fft_ 
		self.e2_fft = e2_fft_ 
		x = tf.multiply(self.r, tf.cast(tf.real(tf.ifft(\
				tf.multiply(tf.conj(self.e1_fft),self.e2_fft))),dtype=tf.float32))
		return x

	def mu_entities(self):
		mu_h_1 = tf.cast(tf.real(tf.ifft(tf.conj(tf.multiply(tf.fft(tf.complex(\
					tf.multiply(self.mu_h, 1./self.r),0.0)), \
					1./self.e2_fft)))), dtype=tf.float32)
		mu_h_2 = tf.cast(tf.real(tf.ifft(tf.multiply(tf.fft(tf.complex(\
					tf.multiply(self.mu_h, 1./self.r),0.0)), \
					1./tf.conj(self.e1_fft)))), dtype=tf.float32)
		return mu_h_1, mu_h_2




