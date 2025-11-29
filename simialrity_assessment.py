import spacy
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import csv

#load spacy model
nlp=  spacy.load("en_core_web_sm")

#initialising
corpus_directory = 'corpus'
documents = []
files = []

#now to gather a list of the files and also their contents
for file in os.listdir(corpus_directory):
	#open in read mode and  save contents to list
	with open (os.path.join(corpus_directory,file),'r') as f:
		contents= f.read()
	documents.append(contents)
	files.append(file)


#TF-IDF 
vectoriser = TfidfVectorizer(stop_words='english',ngram_range = (1,2),max_features =5000, min_df=2,max_df=0.8,norm = "l2")

#apply to corpus and transform into tf-idf matrix
tfidf_matrix = vectoriser.fit_transform(documents)
feature_names= vectoriser.get_feature_names_out()
print('tf-idf matrix shape:',tfidf_matrix.shape)

#bag of words
bag_words = CountVectorizer()
bg_matrix = bag_words.fit_transform(documents)

#embeddings using spacy nlp
embeddings = np.array([nlp(doc).vector for doc in documents])


#function to create the similarity results
def sim_search(query,corpus):
	results = {}
	
	#tfidf similarity
	q_tfidf = vectoriser.transform([query])
	similarities_count = cosine_similarity(q_tfidf, tfidf_matrix).flatten()
	results['tfidf'] = {'Most Similar': corpus[np.argmax(similarities_count)],
			    'Least Similar': corpus[np.argmin(similarities_count)],
			     'Scores': similarities_count}
	
	#bag_of_words similarity
	q_count =  bag_words .transform([query])
	similarities_count = cosine_similarity(q_count,bg_matrix).flatten()
	results['count'] = {
			'Most Similar' : corpus[np.argmax(similarities_count)],
			'Least Similar' : corpus[np.argmin(similarities_count)],
			'Scores' : similarities_count}
	
	#embeddings similarity
	q_embed = nlp(query).vector.reshape(1,-1)
	similarities_embed = cosine_similarity(q_embed,embeddings).flatten()
	results['embeddings'] = {'Most Similar' : corpus[np.argmax(similarities_embed)],
				'Least Similar' : corpus[np.argmin(similarities_embed)],
				'Scores' : similarities_embed
 				} 
		
	return results

#function to output to df
def csv_output(query,output):
	rows = []
	for type, result in output.items():
		rows.append({'Query':query, "Method type": type,
		"Most Similar" : result["Most Similar"], "Least Similar": result["Least Similar"],
		"Scores": result["Scores"].tolist()})
	return pd.DataFrame(rows)

#create list of  strings for comparison
queries =  ['The cat sat on the mat','World politics is becoming increasingly more unstable','Easter Rising occured in Dublin, Ireland','There are many uses for AI in the workplace, and it is increasingly likely that Agentic AI integration may lead to further job losses',"Some claim that the costs of living in Ireland are increasing at a far more rapid pace than salaries" ]
dfs=[]

#run function  on each individual query
for query in queries:
	output =  sim_search(query,documents)
	df = csv_output(query,output)
	dfs.append(df)

#combine all and output
final_df = pd.concat(dfs,ignore_index = True)
final_df.to_csv('files/similarity_search_api_results.csv',index=False)
print(final_df)

