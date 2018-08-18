import tensorflow as tf
import numpy as np
from ..base import KBEModel
no_zeros = 10e-10


class model(KBEModel):
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=5, 
		relation_dim=5,
		lambda_=None,
		lrate=.001,
		model_dir='model' ):
		assert n_entity and n_relation
		name = 'HTPR%ieD%irD%sL' % (entity_dim, relation_dim, str(lambda_) if lambda_ else 'inf')
		h_dim = relation_dim * entity_dim**2
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
		e1_r_e2_tpr = tf.einsum('bni,bnj,bnk->bnijk', self.e1, self.r, self.e2)
		batchdim = tf.cast(tf.shape(self.e1_choice)[0], tf.int32)
		negdim = tf.cast(tf.shape(self.e1_choice)[1], tf.int32)
		flattened_tpr = tf.reshape(tf.reshape(e1_r_e2_tpr, [batchdim, negdim, self.entity_dim, \
						self.relation_dim*self.entity_dim] ), [batchdim, negdim, self.h_dim])
		return flattened_tpr


