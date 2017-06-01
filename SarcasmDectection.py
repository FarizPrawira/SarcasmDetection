import re
import pickle

import numpy as np

__all__ = ['BouaziziOtsuki']

class SarcasmDetection(object):
	"""Base class for Sarcasm Detection
	For now there only one sarcasm detection
	that is the proposed method from Bouazizi and Otsuki.
	"""
	
class BouaziziOtsuki(SarcasmDetection):
	"""
	Proposed method from Bouazizi and Otsuki
	to detect sarcasm on twitter.
	For details on the algorithm:
		http://ieeexplore.ieee.org/document/7417640/

	Paramater
	=========
	preprocess : 
	postag : string ('filename')
		Name of the .txt file for the postager.
		Locate in the assets/'filename'.text
	posWords : string ('filename')
		Name of the .txt file for the list of postive words.
		Locate in the assets/'filename'.text
	negWords : string ('filename')
		Name of the .txt file for the list of negative words.
		Locate in the assets/'filename'.text
	posHashtags : string ('filename')
		Name of the .txt file for the list of postive hashtags.
		Locate in the assets/'filename'.text
	negHashtags : string ('filename')
		Name of the .txt file for the list of negative hashtag.
		Locate in the assets/'filename'.text
	uncomWords : string ('filename')
		Name of the .txt file for the list of uncommon words.
		Locate in the assets/'filename'.text
	patternSarcasm : string ('filename')
		Name of the .txt file for the list of postive words.
		Locate in the assets/'filename'.text

	"""

	def __init__(	self, preprocess='default', postag='POSTAGGER', 
								posWords='kataPositif', negWords='kataNegatif',
								posHashtags='hashtagPositif', negHashtags='hashtagNegatif',
								uncomWords='uncomWords', patternSarcasm='patternSarcasm'):
		# super(BouaziziOtsuki, self).__init__()
		self.preprocess = preprocess
		self.postag = self._loadAssets(postag, pickle_ = True)
		self.posWords = self._loadAssets(posWords)
		self.negWords = self._loadAssets(negWords)
		self.posHashtags = self._loadAssets(posHashtags)
		self.negHashtags = self._loadAssets(negHashtags)
		self.uncomWords = self._loadAssets(uncomWords)
		self.patternSarcasm = self._loadAssets(patternSarcasm)

	def fit(self, raw_documents, y=None):
		""" Learn a sentence from the data trainings in the raw documents.

		Paramater
		=========
		raw_documents : iterable
			an iterable which yields either str, unicode or file objects

		Return
		======
		self
		"""	
		self.fit_transform(raw_documents)
		return self

	def fit_transform(self, raw_documents, y=None):
		""" Learn a sentence from the data trainings and return matrix features extraction.
		This is equivalent to fit followed by transform, but more efficiently
		implemented.

		Paramater
		=========
		raw_documents : iterable
			an iterable which yields either str, unicode or file objects

		Return
		======
		X : sparse matrix, [n_samples, n_features]
			features extraction matrix.
		"""	
		if isinstance(raw_documents, str):
			raise ValueError("Iterable over raw documents expected, "
											"string object received.")
		X = self._getFeatures(raw_documents)
		return X

	def transform(self, raw_documents, copy=True):
		"""Transform documents to document-term matrix.
		Uses the vocabulary and document frequencies (df) learned by fit (or
		fit_transform).

		Parameters
		==========
		raw_documents : iterable
			an iterable which yields either str, unicode or file objects
		copy : boolean, default True
			Whether to copy X and operate on the copy or perform in-place
			operations.

		Returns
		=======
		X : sparse matrix, [n_samples, n_features]
			features extraction matrix.
		"""
		if not hasattr(self, 'fit'):
			raise TypeError("This instance is not fitted yet. Call 'fit' with "
											"appropriate arguments before using this method.")
		if isinstance(raw_documents, str):
			raise ValueError("Iterable over raw documents expected, "
											"string object received.")
		X = self._getFeatures(raw_documents)
		return X

	def _loadAssets(self, assets, pickle_=False):
		if pickle_:
			return pickle.load(open('assets/%s.p' %assets,'rb'))
		return open('assets/%s.txt' %assets,'r').read().split("\n")

	def _findFeatures(self, regex, text):
		return len(regex.findall(text))

	def _postagText(self, text):
		GFI = ["CD", "FW", "NN", "PRP", "SYM", "UH", "MD", "RB", "WDT", "WP"]
		for word, tag in self.postag.tag(text.split()):
			if tag in GFI:
				text = text.replace(word, tag, 1)
		return text

	def _ngram(self, text, n):
		words = text.split()
		ngram = []
		for word in zip(*[words[i:] for i in range(n)]):
			ngram.append(" ".join(word))
		return ngram

	def _sentimentFeatures(self, text):
		# Sentimen Contrast
		textTag = self.postag.tag(text.split())
		emotionalTag = ["JJ", "RB", "VB"]

		pw, nw, PW, NW = (0 for i in range(4))
		for word, tag in textTag:
			if word in self.posWords: 
				pw += 1
				if (tag in emotionalTag): PW += 1
			elif word in self.negWords:
				nw += 1
				if (tag in emotionalTag): NW += 1

		if (not NW and not PW): 
			sentiContrast = 0
		else:
			EWW = 3
			numerator = (EWW*PW + pw) - (EWW*NW + nw)
			denominator = (EWW*PW + pw) + (EWW*NW + nw)
			sentiContrast = numerator / denominator

		# Number of emoticon
		emoticonList = [
			re.compile(r'([:;Xx8=]-?\'?[DPpb3)}\]>])'),	# positive emoticon regex
			re.compile(r'(([;:>]\'?:?-?[@(\\\/\|]))'),	# negative emoticon regex
			re.compile(r'([:;]-?[)pP])') ]							# sarcasm emoticon regex	
		emoticon = [self._findFeatures(emoticon, text) for emoticon in emoticonList]

		# Number of Hashtag
		hashtagWord = [ word for word in text.split() if word.startswith("#") ]
		numPosHashtag, numNegHashtag = (0 for i in range(2))
		for word in hashtagWord:
			if word in self.posHashtags: numPosHashtag +=1
			if word in self.negHashtags: numNegHashtag +=1

		# Coexistence between component 
		WW = True if (pw and nw) else False 
		HH = True if (numPosHashtag and numNegHashtag) else False
		WH = True if ((pw and numNegHashtag) or (nw and numPosHashtag)) else False
		WE = True if ((pw and emoticon[1]) or (nw and emoticon[0])) else False

		return emoticon + [sentiContrast, numPosHashtag, numNegHashtag, WW, HH, WH, WE]

	def _punctuationFeatures(self, text):
		punctuationList = [
			re.compile(r'[!]'),	# Exclamation mark
			re.compile(r'[?]'),	# Question mark
			re.compile(r'[.]'),	# Dots
			re.compile(r'\b[A-Z]{2,}\b'), 				# All Capital Words
			re.compile(r'\"[^\"]+\"|\'[^\']+\''),	# Quoted words
			re.compile(r'([aiueoAIUEO])\1{3,}')] 	# Repeated vowel
		punctuation = [self._findFeatures(punctuation, text) for punctuation in punctuationList]
		punctuation[-1] = bool(punctuation[-1])
		return punctuation

	def _lexicalSyntacticFeatures(self, text):
		# Existance and number of uncommon words
		numUncomWords = len([word for word in text.split() if word in self.uncomWords])

		# Existance of common sarcasm expression
		postagText = self._postagText(text)
		for pattern in self.patternSarcasm:
			comSarExpres = True if pattern in postagText else False
				
		# Number of interjection
		listInterjection = [
			re.compile(r'\b(a+|i+|u+|e+|o+)(h+|w+)\b'), # ah, ih, uh, eh, oh, aw, iw, uw, ew, ow
			re.compile(r'\bw(a+|o+)(w+|h+)\b'), # wow, waw, woh, wah
			re.compile(r'\b(c|i?d|a+)i+h+\b'), # cih, idih, dih, aih
			re.compile(r'\b(d|h)u+h+\b'), # duh, huh
			re.compile(r'\b(h+|u+)m+\b'), # hm, um
			re.compile(r'\bu+p+s+\b'), # ups
			re.compile(r'\bci+e+\b'), # cie
			re.compile(r'\blh+o+\b'), # lho
			re.compile(r'\bpf+t+\b'), # pft
			re.compile(r'\bo+\b')] # o
		numInterjection = sum([self._findFeatures(interjection, text) for interjection in listInterjection])

		# Number of laughter
		listLaughter = [
			re.compile(r'\b(h+a+)+\b|\b(h+i+)+\b|\b(h+e+)+\b|\b(h+o+)+\b'), # haha hihi hehe hoho
			re.compile(r'\bl+o+l+[ol]*\b'),	# lol
			re.compile(r'\b(w+k+)+w*\b')]	# wkwk
		numLaughter = sum([self._findFeatures(laughter, text) for laughter in listLaughter])

		return [bool(numUncomWords), numUncomWords, comSarExpres, numInterjection, numLaughter]

	def _patternFeatures(self, text):
		patternList = {}
		for pattern in self.patternSarcasm:
			patternLen = len(pattern.split())
			patternList.setdefault(patternLen, [])
			patternList[patternLen].append(pattern)

		postagText = self._postagText(text)
		fitur = []
		alpha = 0.03
		for idx, patterns in patternList.items():
			beta = (idx-1)/(idx+1)
			res = 0
			for pattern in patterns:
				if pattern in postagText:
					res += 1
				else:
					n = 0
					for x in range(2, idx):
						ngramList = self._ngram(pattern, x)
						for ngram in ngramList:
							if ngram in postagText:
								n = x
					if n:
						res += alpha*n/idx
			fitur.append(beta*res)
		return fitur

	def _getFeatures(self, raw_documents):
		features = []
		for doc in raw_documents:
			sentiment = self._sentimentFeatures(doc.lower())
			punctuation = self._punctuationFeatures(doc)
			lexicalSyntactic = self._lexicalSyntacticFeatures(doc.lower())
			pattern = self._patternFeatures(doc.lower())
			features.append(sentiment + punctuation + lexicalSyntactic + pattern)
		featuresNP = np.array(features)
		return featuresNP
