import os

def permuteList(list_, permutation):
	return [ list_[i] for i in permutation ]

class defaultFalseDict(dict):
    def __missing__(self,key):
        return False

class defaultTrueDict(dict):
    def __missing__(self,key):
        return True

class defaultValueDict(dict):
    def set_default(self,default_value):
        setattr(self,'default_value',default_value)
    def __missing__(self,key):
        return self.default_value

def readTripletsData(dataDirectory, typed=False):
	"""Read in Knowledge Base embedding data. Expects id files for entities and relations, type constraints for relations, and training, validation, and test data."""
	if not dataDirectory.endswith('/'): dataDirectory = dataDirectory + '/'
	entityFile = 'entity2id.txt'; relationFile = 'relation2id.txt'; typeFile = 'type2id.txt'
	if all( 'typed_' + dtype + '2id.txt' in os.listdir(dataDirectory) for dtype in ['train','valid','test']):
		trainFile = 'typed_train2id.txt'; devFile = 'typed_valid2id.txt'; testFile = 'typed_test2id.txt'
	else: 
		trainFile = 'train2id.txt'; devFile = 'valid2id.txt'; testFile = 'test2id.txt'
	typeConstraintsFile = 'type_constrain.txt'
	#build entity & relation index lookup
	#assumes file2idx.txt has numlines as the first line, and format name\tidx in rest
	with open(dataDirectory + entityFile, 'r') as f:
		entity2idx = { line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:] }
	with open(dataDirectory + relationFile, 'r') as f:
		relation2idx = { line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:] }
	if typeFile in os.listdir(dataDirectory):
		with open(dataDirectory + typeFile, 'r') as f:
			type2idx = { line.split()[0]: int(line.split('\t')[1]) for line in f.readlines()[1:] }
	with open(dataDirectory + trainFile, 'r') as f:
		trainData = [ [ int(i) for i in line.split() ] for line in f.readlines()[1:] ]
	with open(dataDirectory + devFile, 'r') as f:
		devData = [ [ int(i) for i in line.split() ] for line in f.readlines()[1:] ]
	with open(dataDirectory + testFile, 'r') as f:
		testData = [ [ int(i) for i in line.split() ] for line in f.readlines()[1:] ]
	if typeConstraintsFile in os.listdir(dataDirectory):
		with open(dataDirectory + typeConstraintsFile, 'r') as f:
			lines = f.readlines()
		typeConstraints_left = { line[0]: line[2:] for line in [ [ int(i) for i in \
				l.split() ] for j, l in enumerate(lines[1:]) if j%2 == 0 ] }
		typeConstraints_right = { line[0]: line[2:] for line in [ [ int(i) for i in \
				l.split() ] for j, l in enumerate(lines[1:]) if j%2 == 1 ] }
	else: 
		typeConstraints_left = defaultValueDict()
		typeConstraints_left.set_default(list(range(len(entityIndexLookup))))
		typeConstraints_right = defaultValueDict()
		typeConstraints_right.set_default(list(range(len(entityIndexLookup))))
	#compile triplets filter -- returns True if the triplet occurs in the training, dev, or test data. 
	tripletsFilter = defaultFalseDict({(line[0], line[2], line[1]): True for line in trainData + devData + testData} )
	e1s_train = []; rs_train = []; e2s_train = []
	if typed:
		e1types_train = []
	for line in trainData:
		e1s_train.append(line[0]); rs_train.append(line[2]); e2s_train.append(line[1]) 
		if typed: e1types_train.append(line[3])
	trainData = [e1s_train, rs_train, e2s_train]
	if typed: trainData.append(e1types_train)
	e1s_dev = []; rs_dev = []; e2s_dev = []
	if typed: e1types_dev = []
	for line in devData:
		e1s_dev.append(line[0]); rs_dev.append(line[2]); e2s_dev.append(line[1])
		if typed: e1types_dev.append(line[3])
	devData = [e1s_dev, rs_dev, e2s_dev]
	if typed: devData.append(e1types_dev)
	e1s_test = []; rs_test = []; e2s_test = []
	if typed: e1types_test = []
	for line in testData:
		e1s_test.append(line[0]); rs_test.append(line[2]); e2s_test.append(line[1])
		if typed: e1types_test.append(line[3])
	testData = [e1s_test, rs_test, e2s_test]
	if typed: testData.append(e1types_test)
	data = {'train':trainData, 'valid':devData, 'test':testData, 'filter':tripletsFilter, 'entity2idx':entity2idx,
			'relation2idx':relation2idx, 'candidates_l':typeConstraints_left, 'candidates_r':typeConstraints_right}
	if typed:
		data['type2idx'] = type2idx
	else:
		data['train'], data['valid'], data['test'] = [ D[:3] for D in [trainData, devData, testData] ]	#remove type info if it is there
	return data
		


def build_namedict(nameFile):
	with open(nameFile, 'r') as f:
		l = f.readlines()
	D = collections.defaultdict(list)
	for line in l:
		mid, name = [ o.strip() for o in line.split('\t')]
		D[mid].append(name)
	return dict(D)




