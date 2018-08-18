1443-Compounds Dataset 2.0

(1) SOURCE

The file 1443_Compounds.txt contains 1,443 noun-noun compounds extracted from the British National Corpus (BNC). Each compound is annotated with semantic relations at three levels of granularity. The construction of the dataset is described in the following paper:

Diarmuid Ó Séaghdha. 2007. Designing and Evaluating a Semantic Annotation Scheme for Compound Nouns. In Proceedings of Corpus Linguistics 2007. (Available online at http://www.cl.cam.ac.uk/~do242/Papers/dos_cl2007.pdf)

The annotation guidelines used are available at http://www.cl.cam.ac.uk/~do242/guidelines.pdf

Each line of the file consists of five space-separated columns:

1: The compound modifer
2: The compound head
3: One of six coarse relations: BE, HAVE, IN, ACTOR, INST and ABOUT. 
4: A direction indicator, either 1 or 2, describing which of the head and modifier occupy the first argument slot for the semantic relation. For example, "fruit dish" and "dish fruit" would both be labelled IN, but "fruit dish" has direction 1 and "dish fruit" would have direction 2. BE is assumed to be symmetric, so when column 3 is 'BE' this column is always 1.
5: A fine-grained relations corresponding to a rule in the annotation guidelines document

This multi-level annotation suggests three experimental designs with different sets of class labels:

COARSE - column 3 only
DIRECTED - columns 3 and 4
FINE - columns 4 and 5

The file 1443_Compounds.cv additionally contains information about the cross-validation folds used for classification by Ó Séaghdha and Copestake (2007, 2008):

Diarmuid Ó Séaghdha and Ann Copestake. 2007. Co-occurrence Contexts for Noun Compound Interpretation. In Proceedings of the ACL-07 Workshop A Broader Perspective on Multiword Expressions. (Available online at http://acl.ldc.upenn.edu/W/W07/W07-1108.pdf)

Diarmuid Ó Séaghdha and Ann Copestake. 2008. Distributional kernels for semantic classification. In Proceedings of the 22nd International Conference on Computational Linguistics (COLING 2008). (Available online at http://www.cl.cam.ac.uk/~do242/Papers/Coling08.pdf)


(2) TERMS OF USE

The annotated data is Copyright Diarmuid Ó Séaghdha, 2008. Licensed under the Creative Commons Attribution-Share Alike 3.0 Unported license (http://creativecommons.org/licenses/by-sa/3.0/). Some rights reserved.

The data may be used for any purpose, provided that the following paper is cited in any publication:

Diarmuid Ó Séaghdha and Ann Copestake. 2007. Co-occurrence Contexts for Noun Compound Interpretation. In Proceedings of the ACL-07 Workshop A Broader Perspective on Multiword Expressions.

The data may be redistributed, so long as an identical license is used.


(3) CONTACT

Any questions about the data should be addressed to Diarmuid Ó Séaghdha (Diarmuid.O'Seaghdha AT cl.cam.ac.uk)
