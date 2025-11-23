import spacy
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
nlp=  spacy.load("en_core_web_sm")

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

word_scores = tfidf_matrix.sum(axis=0).A1
word_dict = dict(zip(feature_names,word_scores))

wordcloud =  WordCloud(width=1000,height=500,background_color ='white')
wordcloud.generate_from_frequencies(word_dict)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.savefig('images/wordcloud.png')

def term_per_document(tfidf_matrix,feature_names, top_n_terms=5, filename='terms_per_doc.csv'):
	with open(filename,"w", newline="") as f:
		for document_id in range(tfidf_matrix.shape[0]):
			row = tfidf_matrix[document_id].toarray().flatten()
			top_id = row. argsort()[::-1][:top_n_terms]
			for i in  top_id:
				if row[i] > 0:														
					writer.writerow([document_id+1,feature_names[i],round(row[i],3)])
term_per_document(tfidf_matrix,feature_names,top_n_terms=5)
