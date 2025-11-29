import spacy
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import csv
#load up the spacy model
nlp=  spacy.load("en_core_web_sm")

#initialisation
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

entities = []
print(files)
#now run spacy on the contents
for i, document_text  in enumerate(documents):
	print(f"processing document: {files[i]}")
	doc = nlp(document_text)
	#create a tuple that contins the   entity, label pairs
	ents = [(entity.text,entity.label_) for entity in doc.ents] 
	for entity_text, entity_label in ents:
		#print(entity_text,entity_label)
		entities.append({'filename': files[i],
				'entity_text':entity_text,
				'entity_label':entity_label})

#descriptive analytics

#label chart
df = pd.DataFrame(entities)
label_counts = df.entity_label.value_counts().reset_index() # was series, need df
label_counts.columns =['entity_label','count'] 
sns.barplot(data=label_counts, x = 'entity_label',y='count')
plt.title("Label Distribution")
plt.xlabel('Label')
plt.xticks(rotation=90)
plt.ylabel('Count')
plt.savefig("images/entity_label_counts.png")

#text chart
text_counts = df.entity_text.value_counts().reset_index().head(20) # was series, need top 20 df
text_counts.columns = ['entity_text','counts']
sns.barplot(data=text_counts,x='entity_text',y='counts')
plt.title('Text Distribution')
plt.xlabel('Text')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.savefig('images/entity_text_counts.png')

#entitines per file
plt.figure(3)
entities_count =  df.filename.value_counts().reset_index() #was series, need df
entities_count.columns = ['filename','entity_count']
sns.histplot(data=entities_count , x = 'entity_count',bins=10,kde=True)
plt.title('Distribution of Named Entities per document')
plt.xlabel('# named entities')
plt.ylabel('Document')
plt.xticks(rotation = 90)
plt.savefig('images/entity_count_distribution.png')
plt.show()

#TF-IDF 
vectoriser = TfidfVectorizer(stop_words='english',ngram_range = (1,2),max_features =5000, min_df=2,max_df=0.8,norm = "l2")

#apply to corpus and transform into tf-idf matrix
tfidf_matrix = vectoriser.fit_transform(documents)
feature_names= vectoriser.get_feature_names_out()

print('tf-idf matrix shape:',tfidf_matrix.shape)

#generate a wordcloud from the tfidf matrix
word_scores = tfidf_matrix.sum(axis=0).A1
word_dict = dict(zip(feature_names,word_scores))

wordcloud =  WordCloud(width=1000,height=500,background_color ='white')
wordcloud.generate_from_frequencies(word_dict)

#plot wordcloud
plt.figure(figsize=(10,5))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.savefig('images/wordcloud.png')

#function to find the top terms in each document
def term_per_document(tfidf_matrix,feature_names, top_n_terms=5, filename='terms_per_doc.csv'):
	with open(filename,"w", newline="") as f:
		writer =csv.writer(f)
		#columns - file, the term and the score
		writer.writerow(['Document','Term','Score'])	
		for document_id in range(tfidf_matrix.shape[0]):
			row = tfidf_matrix[document_id].toarray().flatten()
			top_id = row. argsort()[::-1][:top_n_terms]
			for i in  top_id:
				if row[i] > 0:														
					writer.writerow([document_id+1,feature_names[i],round(row[i],3)])
term_per_document(tfidf_matrix,feature_names,top_n_terms=5)
