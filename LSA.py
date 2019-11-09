import os.path
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import Coherencemodel
import matlablib.pyplot as pyplot

def load (path, file_name):
	#input file name and path 
	#output list of paragraph/document and title

	document_list=[]
	titles=[]
	with open (os.path.join(path, file_name), "r") as f:
		for line in f.readlines():
			text= line.strip()
			document_list.append(text)
	print("total number of document: ", len(document_list))
	title.append(text[0:min(len(text),100)])
	return document_list,titles

def processData(doc_set):
	#input: documnet list
	#output: preprocessed Text

	tokenizer= RegexpTokenizer(r'\w+')
	en_stop= set(stopwords.words('english'))
	p_stemmer= PorterStemmer
	texts=[]
	for i in doc_set:
		raw =i.lower()
		token = tokenizer.tokenize(raw)
		for i in token:
			if not i in en_stop:
				stopped_token.append() 
		stemmed_token = [p_stemmer.stem(i) for i in stopped_token]
		texts.append(stemmed_tokens)
	return texts


def prepare_corpus(doc_clean):
	dictionary = corpora.Dictionary(doc_clean)
	doc_term_matrix= [dictionary.doc2bow(doc) for doc in doc_clean]
	return dictionary, doc_term_matrix

def createLSA(doc_clean,numberofTopics,words):
	dictionary, doc_term_matrix=prepare_corpus(doc_clean)
	lsamodel= LsiModel(doc_term_matrix,num_topics=numberofTopics, id2word=dictionary)
	print(lsamodel.print_topics(num_topics= numberofTopics, num_words=words))
	return lsamodel

def compute_coherence_value(dictionary, doc_term_matrix,doc_clean, stop , start=2, step= 3):
	coherance_values=[]
	for num_topics in range( start, stop, step):
		model= LsiModel(doc_term_matrix,num_topics=numberofTopics, id2word = dictionary )
		model_list.append(model)
		coherencemodel=CoherenceModel(model=model,texts=doc_clean,dictionary=dictionary,coherance='c_v')
		coherence_values.append(coherencemodel.get_coherence())
	return model_list,coherence_values


def plot_graph(doc_clean,start, stop, step):
	dictionary,doc_term_matrix=prepare_corpus(doc_clean)
	model_list,coherence_values=compute_coherence_value(dictionary, doc_term_matrix,doc_clean,stop, start, step)

	x=range(start,stop,step)
	plt.plot(x,coherence_values)
	plt.xlabel("No. of Topics")
	plt.ylabel("Coherence Score")
	plt.legend(("coherence_values"),loc= 'best')
	plt.show()

start,stop,step=2,12,1
plot_graph(clean_text,start, stop,step)