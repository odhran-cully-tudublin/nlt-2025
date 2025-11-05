from langchain_community.llms import Ollama
import os

llm =  Ollama(model="gemma:2b")
topics = ["The role of AI in HealthCare","The future of transport",
	"The importance of renewables in the 21st century", "The dangers of social media and companies' ability to overlook them",
	 "The effects of climate change on migration", "A short summary Irish history and important political figures",
	 "Write a news article on the rise of right wing extremism", "American political scene",
	"Cat behaviour and our lack of understanding of them","The growing need for cybersecurity" ]
os.makedirs("corpus",exist_ok=True)

for i, topic in enumerate(topics):
	print(f"Document {i} on topic: {topic}")
	response=llm.invoke(f"Write 500 words on {topic}. Ensure the presence of Named entities for Named entity recognition.")
	with open(f"corpus/doc_{i+1}.txt","w") as f:
		f.write(response)

