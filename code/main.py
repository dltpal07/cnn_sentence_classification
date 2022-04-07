#############################################
#  											#	
#  These codes are just an example to you.	#
#  You can modify these codes.				#
#  It is not necessary to use them.			#
#											#
#############################################
import utils


# load data
tr_sents, tr_labels = utils.load_data(filepath='../data/sent_class.train.csv')
ts_sents, ts_labels = utils.load_data(filepath='../data/sent_class.test.csv')

# tokenization
tr_tokens = utils.tokenization(tr_sents)
ts_tokens = utils.tokenization(ts_sents)

# lemmatization
tr_lemmas = utils.lemmatization(tr_tokens)
ts_lemmas = utils.lemmatization(ts_tokens)

# character one-hot representation
tr_char_onehot = utils.char_onehot(tr_lemmas)
ts_char_onehot = utils.char_onehot(ts_lemmas)

# sequence classification

