import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10


class model(KBEModel):
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=50, 
		lambda_=None,
		lrate=.001,
		model_dir='model' ):
		assert n_entity and n_relation
		name = 'HHolE%iD%sL' % (entity_dim, str(lambda_) if lambda_ else 'inf')
		relation_dim = h_dim = entity_dim
		super(model, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					lambda_=None,
					lrate=lrate,
					name=name,
					model_dir=model_dir )


	def build_x(self):
		self.e1_fft = tf.fft(tf.complex(self.e1,0.0))
		self.e2_fft_ = tf.fft(tf.complex(self.e2,0.0)); 
		self.e2_fft = self.e2_fft_ + tf.complex(tf.to_float(tf.equal(self.e2_fft_, 0.)),0.)*no_zeros 
		x = tf.multiply(self.r, tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(self.e1_fft),self.e2_fft))),dtype=tf.float32))
		return x


