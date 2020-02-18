import argparse

def get_args():
	    
	parser = argparse.ArgumentParser( description = __doc__ )

	# Configuration file path argument.
	parser.add_argument(
						'-c', '--config',
						metavar = 'C',
						help = 'The Configuration file',
						default = './configuration_files/cifar_config.json',
						required = False
					)

	# Epoch size argument.
	parser.add_argument(
						'-e', '--epoch',
						metavar = 'E',
						help = 'Number of epoches for traning the model',
						default = 1,
						required = False
					)

	# Convert to dictonary.
	args = vars(parser.parse_args())

	if( args['config'] == './configuration_files/fashion_config.json' ):
		print('Using default configuration file.')
	else:
		print('Using configurations from file:', args['config'])

	if( args['epoch'] == 1 ):
		print('Using default epoch size of 1.')
	else:
		print('Using epoch size:', args['epoch'])
		
	return args
