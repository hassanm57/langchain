from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=500)
prompt = "Hello! How can I assist you today? What would you like to know or discuss?"

response = model.invoke(prompt)
print(response.content)