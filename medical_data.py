import  pandas as pd
import re
import spacy
from langchain_ollama import OllamaLLM as  Ollama
patients = pd.read_csv('patients.csv')
patients.columns=patients.columns.str.lower()
dictionary = pd.read_csv('index.csv')
llm = Ollama(model='gemma:2b')
cat_cols = ['smoker_status','riluzole_use','c9orf72_mutation','cognitive_impairment','family_history']
patients[cat_cols]= patients[cat_cols].astype("category")

#print(patients.Survival_Time_mo.value_counts())
#print(patients.dtypes)

def llm_query (prompt,context=""):
	full_prompt = f"User query: {prompt} \n\nContext:\n{context}\n\nAnswer clearly:"
	response = llm.invoke(full_prompt)
	return response

#Tools

def lookup(feature_name):
	entry =  dictionary[dictionary['Variable Name'].str.lower() == feature_name]
	if entry.empty:
		return f"No dictionary entry for {feature_name}"
	else:
		decription = entry.Description.values[0]
		notes = entry['Data Type / Range / Notes'].values[0]
		return f"{feature_name}: {decription}: (Notes: {notes})"

def statistics(feature_name):
	if feature_name not in patients.columns:
		return f"{feature_name} not in patients dataset" 
	column = patients[feature_name]
	dtype = column.dtype
	if dtype in ['int64','float64']:	
		stats = column.describe()
		stats_string =  f"For {feature_name}- numerical - , count is {stats['count']}, the avg is {stats.mean()}, standard deviation is {stats.std()}, the minimum observed value is {stats.min()}, maximum is {stats.max()} "
		return stats_string
	elif dtype in ['object','category']:
		stats = column.describe()
		top_values = column.value_counts().head().to_dict()
		stats_string = f"For {feature_name}- categorical - , count is {stats['count']}, unique number is {stats['unique']}, top is {stats['top']},frequency of {stats['freq']}, value counts head is {top_values}"
		return stats_string
	else:
		return f"{feature_name} is of unsupported dtype {dtype}" 
def correlations(df, target= "survival_time_mo"):
	numeric_cols =  patients.select_dtypes(include=['number'])
	results = []
	for col in numeric_cols.columns:
		correlations = numeric_cols[col].corr(numeric_cols[target])
		results.append( f"Correlation between {col} and  {target} is {correlations}")
		corr_str = "\n".join (results)
	return  corr_str

#Rag Prep

def prep_context(feature_name):
	definition = lookup(feature_name)
	stats = statistics(feature_name)
	return f"Definition: \n{definition}\n\nStats:\n{stats}"

def llm_query_rag(prompt,context=""):
	full_prompt =  f"{prompt}\n\n Here is some helpful information, do not answer without considering the provided information. If that does not contribute an answer and you do not know the naswer, simply inform me that you do not have enough information. \n {context}"

#Basic attempt - feed in single feature 
#query = "How many unique entries in the chosen feature?"
#context =  prep_context('Gender')
#print(context)


# RAG Approach on all features
context_lines=[]
for feature_name in patients.columns:
	summary = prep_context(feature_name)
	context_lines.append(summary)
#join together all items to provide a context spanning all features
rag_context = "\n".join(context_lines)
#print(llm_query(query,rag_context))

#Agentic Approach - Attempt to use nlp to parse the query to determine  what tool is needed
#then have the llm use the tool as needed

nlp = spacy.load('en_core_web_sm')

#Agentic Prep

def extract_features(query,patients):
	doc = nlp(query.lower())
	features = []
	for token in doc:
		if token.text in patients.columns.str.lower().tolist():
			features.append(token.text)
	return list(set(features))

def intent_matching(query):
	query = query.lower()
	if "define" in query or "meaning" in query or "definition" in query or "mean" in query:	
		return  "definition"
	elif "stat" in query or "stats" in query or  "distribution" in query or "average" in query or "statistics" in query :
		return "statistics"
	elif "comparison" in query or  "compare" in query or "versus" in query or "vs" in query:
		return "compare"
	elif "relationship" in query or  "predict" in query or "correlate" in query or "correlation" in query:
		return "correlation"
	else:
		return  "general query"

def agent_1(query):
	intention = intent_matching(query)
	features = extract_features(query,patients)
	print(f"Intention found: {intention} and features found: {features}")
	if intention == "definition" and features:
		agent_context = "\n".join([lookup(feat) for feat in features])
	elif intention == "statistics" and features:
		agent_context = "\n".join([statistics(feat) for feat in features])
	elif intention == "correlation":
		agent_context = correlations(patients)
	elif intention ==  "compare" and len(features) >= 2:
		agent_context = statistics(features[0])+ "\n\n" + statistics(features[1])
	else:
		agent_context = "No tool found" 
	print(agent_context)
	return llm.invoke(f"{query}\n\n . Relevant information is shown below: \n\n {agent_context}")


#Run agent with rule  based intent matching 
#query =  "Define Gender"
#print(agent_1(query))

#Autonomous agent with  own intent matching
def agent_2(query,patients):
	decision_prompt=f"""
			You are an AI agent who's role is to decide which tool to use for a query.
		Tools available are: 
		- definition - explain what a feature is
		- statistics - provide a statistical summary of a feature
		- compare - use where more than one feature is mentioned 
		- correlation - find relationships between features
		- general (only if none of the above apply)
		Query is: {query}
		Use the above descriptions to determine the most similar based on the query. Do not add any punctuation, explanation or extra text to your answer."""

	intention = llm.invoke(decision_prompt).strip().lower()
	valid_tools = ['definition','statistics','compare','correlation','general']

	for word in intention.split():
		word_clean = re.sub(r'[^a-z]','',word)
		if word_clean in valid_tools:
			intention = word_clean
			break
		else: 
			intention ='general'
	features = extract_features(query,patients)	
	print(f"LLM defined intention: {intention} with features {features}")
	if intention == "definition" and features:
                agent_context = "\n".join([lookup(feat) for feat in features])
	elif intention == "statistics" and features:
                agent_context = "\n".join([statistics(feat) for feat in features])
	elif intention == "correlation":
                agent_context = correlations(patients)
	elif intention ==  "compare" and len(features) >= 2:
                agent_context = statistics(features[0])+ "\n\n" + statistics(features[1])
	else:
                agent_context = "No tool found"	
	final_prompt = f"{query}. \n\nAnswer using the context below. Do not answer without using the provided context. If the provided context does not help, say you do not have an answer. \n\n {agent_context}"
	print(final_prompt)
	return llm.invoke(final_prompt)


query = input("Enter Query:")
print(agent_2(query,patients))
 
