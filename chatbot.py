from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(model_name="gpt-4", temperature=0.8)
prompt = "Hello! How can I assist you today?"

llm = model.generate([prompt])
print(llm[0].text)