import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10


class HHolE(KBEModel):
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=50, 
		lambda_=None,
		lrate=.001,
		model_dir='trained_models',
		dataName='DataUnknown',
		epoch_num=None ):
		assert n_entity and n_relation
		name = 'HHolE%iD%sL.%s' % (entity_dim, str(lambda_) if lambda_ else 'inf', dataName)
		relation_dim = h_dim = entity_dim
		super(HHolE, self).__init__(entity_dim=entity_dim,
					relation_dim=relation_dim,
					h_dim=h_dim,
					n_entity=n_entity,
					n_relation=n_relation,
					lambda_=lambda_,
					lrate=lrate,
					name=name,
					model_dir=model_dir,
					epoch_num=epoch_num)
		self.mu_h_1, self.mu_h_2 = self.mu_entities()


	def build_x(self):
		self.e1_fft = tf.fft(tf.complex(self.e1,0.0))
		self.e2_fft_ = tf.fft(tf.complex(self.e2,0.0)); 
		self.e2_fft = self.e2_fft_ + tf.complex(tf.to_float(tf.equal(self.e2_fft_, 0.)),0.)*no_zeros 
		x = tf.multiply(self.r, tf.cast(tf.real(tf.ifft(tf.multiply(tf.conj(self.e1_fft),self.e2_fft))),dtype=tf.float32))
		return x

	def mu_entities(self):
		#compute conditional mu_h1 (optimized e1 in context of r, e2)
		true_e1s = self.e1[:,0,:]
		true_rs = self.r[:,0,:]
		true_e2s = self.e2[:,0,:]
		true_mu_h = self.mu_h[:,0,:]
		mu_h_1 = tf.cast(tf.real(tf.ifft(tf.conj(tf.multiply(tf.fft(tf.complex(true_mu_h,0.0)), \
			1./self.e2_fft[:,0,:])))), dtype=tf.float32)

		mu_h_2 = tf.cast(tf.real(tf.ifft(tf.conj(tf.multiply(tf.fft(tf.complex(true_mu_h,0.0)), \
			1./self.e1_fft[:,0,:])))), dtype=tf.float32)
		return mu_h_1, mu_h_2




