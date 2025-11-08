import spacy
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

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
df = pd.DataFrame(entities)
label_counts = df.entity_label.value_counts().reset_index() # was series, need df
label_counts.columns =['entity_label','count'] 
sns.barplot(data=label_counts, x = 'entity_label',y='count')
plt.title("Label Distribution")
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()
plt.savefig("images/entity_label_counts.png")


text_counts = df.entity_text.value_counts().reset_index().head(20) # was series, need top 20 df
text_counts.columns = ['entity_text','counts']
sns.barplot(data=text_counts,x='entity_text',y='counts')
plt.title('Text Distribution')
plt.xlabel('Text')
plt.ylabel('Count')
plt.show()
plt.savefig('images/entity_text_counts.png')

entities_count =  df.filename.value_counts().reset_index() #was series, need df
entities_count.columns = ['filename','entity_count']

sns.histplot(data=entities_count , x = 'entity_count',bins=10,kde=True)
plt.title('Distribution of Named Entities per document')
plt.xlabel('# named entities')
plt.ylabel('Document')
plt.savefig('images/entity_count_distribution.png')
plt.show()

#TF-IDF 
vectoriser = TfidfVectrozer(stop_words='english',ngram_range = (1,2),max_features =5000, min_df=2,max_df=0.8,norm = "l2")

#apply to corpus and transform into tf-idf matrix
tfidf_matrix = vectoriser.fit_transform(documents)
feature_names= vectoriser.get_feature_names_out()

print('tf-idf matrix shape:',tfidf_matrix.shape)
