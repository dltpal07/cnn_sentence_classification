#############################################
#  											#	
#  These codes are just an example to you.	#
#  You can modify these codes.				#
#  It is not necessary to use them.			#
#											#
#############################################
from typing import Union, List, Dict


def load_data(filepath: str = 'YOUR/CSV/FILE/PATH')
	-> Union[List[str], List[int]]:
	"""
		TO DO: read .csv file and load data
	"""
	return data, targets


def tokenization(sents: List[str]) -> List[List[str]]:
	"""
		TO DO: tokenize sentences into the list of words
	"""
	return tokens


def lemmatization(tokens: List[List[str]]) -> List[List[str]]:
	"""
		TO DO: lemmatize stem from the words
	"""
	return lemmas


def pos_tagging(words: List[str]) -> Dict[str, str]:
	"""
		Use this method when lemmatizing

		Input: list of words
		Output: {word: tag}
	"""
	words_only_alpha = [w for w in words if w.isalpha()]

	def format_conversion(v, pos_tags=['n', 'v', 'a', 'r', 's']):
		w, p = v
		p_lower = p[0].lower()
		p_new = 'n' if p_lower not in pos_tags else p_lower
		return w, p
	
	res_pos = pos_tag(words_only_alpha)
	word2pos = {w:p for w, p in list(map(format_conversion, res_pos))}

	for w in words:
		if w not in word2pos:
			word2pos[w] = 'n'

	return word2pos


def char_onehot(lemmas: List[List[str]]) -> Tensor:
	"""
		TO DO: convert characters in lemmatized word to one-hot vector
	"""
	return v

