#############################################
#  											#	
#  These codes are just an example to you.	#
#  You can modify these codes.				#
#  It is not necessary to use them.			#
#											#
#############################################
from typing import Union, List, Dict
import torch
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

max_len_of_sent = 20
max_len_of_word = 10

def load_data(file_path) -> Union[List[str], List[int]]:
	df = pd.read_csv(file_path)
	data = df['sentence']
	targets = df['label']
	return data, targets


def tokenization(sents: List[str]) -> List[List[str]]:
	tokens = []
	for sent in sents:
		token = word_tokenize(sent)
		tokens.append(token)

	return tokens


def lemmatization(tokens: List[List[str]]) -> List[List[str]]:
	lemmas = []
	lemmatizer = WordNetLemmatizer()
	for token in tokens:
		word2pos = pos_tagging(token)
		lemmas.append([lemmatizer.lemmatize(t, word2pos[t]) for t in token])
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
		return w, p_new

	res_pos = pos_tag(words_only_alpha)

	word2pos = {w:p for w, p in list(map(format_conversion, res_pos))}

	for w in words:
		if w not in word2pos:
			word2pos[w] = 'n'

	return word2pos

def make_char_dict(lemmas):
	char_dict = {}
	char_dict['<P>'] = 0
	char_dict['<U>'] = 1
	sum_lemmas = sum(lemmas, [])
	sum_lemmas = "".join(sum_lemmas)
	sum_lemmas = list(sum_lemmas)
	char_list = list(set(sum_lemmas))
	char_list.sort()
	idx = 2
	for char in char_list:
		char_dict[char] = idx
		idx += 1
	return char_dict


def char_onehot(lemmas, char_dict):
	"""
		TO DO: convert characters in lemmatized word to one-hot vector
	"""
	v = torch.zeros((len(lemmas), max_len_of_sent, max_len_of_word, char_dict.__len__()))
	for lemma_idx, lemma in enumerate(lemmas):
		for word_idx, word in enumerate(lemma):
			if word_idx >= max_len_of_sent:
				break
			for char_idx, char in enumerate(word):
				if char_idx >= max_len_of_word:
					break
				if char in char_dict.keys():
					v[lemma_idx][word_idx][char_idx][char_dict[char]] = 1
				else:
					v[lemma_idx][word_idx][char_idx][char_dict['<U>']] = 1
			if len(word) < max_len_of_word:
				v[lemma_idx,word_idx,len(word):,char_dict['<P>']] = 1
		if len(lemma) < max_len_of_sent:
			tmp_v = torch.zeros((max_len_of_word, char_dict.__len__()))
			tmp_v[:, char_dict['<P>']] = 1
			v[lemma_idx, len(lemma):] = tmp_v

	return v

