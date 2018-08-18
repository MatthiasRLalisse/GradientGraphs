README - gradgraphs models
=============================

A gg model MUST specify a compositional function f_comp(e1,r,e2) for building the compositional embedding x.  build_x should return a single tensorflow object that applies f_comp to the input embeddings.

Optionally, the model may also specify mu_entities, which returns a pair of tensorflow objects obtained by inverting the composition function. 

In some cases, it might also be desirable to override the base class definitions of triplet scores (by default, these are just the Harmony values) and loss (softmax cross entropy). This can be done by overriding the build_scores and build_loss methods for the KBEModel class (see base.py). 

A schematic model specfication file looks like the following:
(make sure to add import statements to models/__init__.py to make new classes visible from imports of gg)

===============================================================================================
#MyModel.py
import tensorflow as tf
import numpy as np
from ..base import KBEModel

class MyModel(KBEModel):
	def __init__(self,
		n_entity=None,		#number of entity embeddings
		n_relation=None,	#number of relation embeddings
		entity_dim=50, 		#dimension of entity embeddings
		relation_dim=50,	#dimension of relation embeddings
		lambda_=None,		#faithfulness penalty
		lrate=.001,		#learning rate
		model_dir='trained_models',	#directory where trained models are stored
		dataName='DataUnknown',		#name of the dataset (appended to the model name when saving)
		epoch_num=None):	#to restore model trained for epoch_num epochs
		assert n_entity and n_relation
		name = 'MyModel%ieD%irD%sL.%s' % (entity_dim, relation_dim, str(lambda_) if \
							lambda_ else 'inf', dataName)	#sets naming convention for saving
											#instances of the model
		h_dim = ???			#dimension of the hidden layer (==dim of x)
		super(NewModel, self).__init__(entity_dim=entity_dim,	#pass the following arguments to the KBEModel superclass.
					relation_dim=relation_dim,	#this class (in base.py) specifies the core network 
					h_dim=h_dim,			#parameters & routines
					n_entity=n_entity,
					n_relation=n_relation,
					lambda_=lambda_,
					lrate=lrate,
					name=name,
					model_dir=model_dir)
		self.mu_h_1, self.mu_h_2 = self.mu_entities()

	def build_x(self):	#the most important part: specify how to build the compositional embedding x
		...
		x = f_comp(self.e1, self.r, self.e2) 	#self.e1,self.r,self.e2 are inherited from the base class
		...
		return x
	
	def mu_entities(self):		#retrieve token embeddings of e1 and e2
		#compute conditional mu_h_1 & mu_h_2  (optimized entity tokens) 
		true_e1s = self.e1[:,0,:]
		true_rs = self.r[:,0,:]
		true_e2s = self.e2[:,0,:]
		true_mu_h = self.mu_h[:,0,:]
		...
		mu_h_1 = f_comp_inv1(true_mu_h, true_rs, true_e2s)
		mu_h_2 = f_comp_inv2(true_mu_h, true_e1s, true_rs)
		...
		return mu_h_1, mu_h_2



