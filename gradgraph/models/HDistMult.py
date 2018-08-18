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
		name = 'DistMult%iD%sL' % (entity_dim, str(lambda_) if lambda_ else 'inf')
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
		x = tf.multiply(self.e1, tf.multiply(self.r,self.e2))
		return x


