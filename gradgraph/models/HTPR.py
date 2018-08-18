import tensorflow as tf
import numpy as np
from ..base import KBEModel

class HTPR(KBEModel):
	def __init__(self,
		n_entity=None,
		n_relation=None,
		entity_dim=5, 
		relation_dim=None,
		lambda_=None,
		lrate=.001,
		model_dir='trained_models',
		dataName='DataUnknown',
		epoch_num=None ):
		if not relation_dim: relation_dim = entity_dim
		name = 'HTPR%ieD%irD%sL.%s' % (entity_dim, relation_dim, str(lambda_) if \
							lambda_ else 'inf', dataName)
		h_dim = relation_dim * entity_dim**2
		super(HTPR, self).__init__(entity_dim=entity_dim,
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
		e1_r_e2_tpr = tf.einsum('bni,bnj,bnk->bnijk', self.e1, self.r, self.e2)
		self.batchdim = tf.cast(tf.shape(self.e1_choice)[0], tf.int32)
		negdim = tf.cast(tf.shape(self.e1_choice)[1], tf.int32)
		ravelled_tpr = tf.reshape(tf.reshape(e1_r_e2_tpr, [self.batchdim, negdim, \
						self.entity_dim, self.relation_dim*self.entity_dim] ), \
						[self.batchdim, negdim, self.h_dim])
		return ravelled_tpr
	
	def mu_entities(self):
		#compute conditional mu_h1 (optimized e1 in context of r, e2)
		true_e1s = self.e1[:,0,:]
		true_rs = self.r[:,0,:]
		true_e2s = self.e2[:,0,:]
		true_mu_h = self.mu_h[:,0,:]
		unravelled_tpr = tf.reshape(tf.reshape(true_mu_h, [self.batchdim, self.entity_dim, \
					self.relation_dim*self.entity_dim]), [self.batchdim, \
					self.entity_dim, self.relation_dim, self.entity_dim])
		mu_h_1 = tf.einsum('bk,bikj,bj->bi', true_rs, unravelled_tpr, true_e2s )
		mu_h_2 = tf.einsum('bi,bk,bikj->bj', true_e1s, true_rs, unravelled_tpr )
		return mu_h_1, mu_h_2
	


