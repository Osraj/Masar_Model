
"""
# what is in this version of the code
added the ability to see past messages from the same conversation
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install --upgrade langchain langchain-community langchain-chroma
# %pip install -qU langchain-groq
# %pip install langchain_openai
# %pip install --upgrade langchain_huggingface
# %pip install --upgrade unstructured openpyxl
# %pip install nltk
# %pip install --upgrade --quiet  langchain sentence_transformers
# %pip install xlrd
# %pip install xformers
# ! pip install einops
# ! pip install transformers
# ! pip install -U sentence-transformers
# ! pip install chromadb

"""## Imports"""

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import nltk
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import JinaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import trim_messages, AIMessage, HumanMessage

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

"""## Data Indexing

### 1.Data Loading

#### txt
"""

loader = DirectoryLoader("../Data/", glob="*/*.txt")
docs = loader.load()

docs

"""### 2.Data Splitting"""

# split the doc into smaller chunks i.e. chunk_size=512
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_documents(docs)

# Fixing the metadata if something is wrong with it
for chunk in chunks:
    for key, value in chunk.metadata.items():
        if isinstance(value, list):
            chunk.metadata[key] = ','.join(value)  # Convert list to a comma-separated string

chunks[5].page_content

"""### 3.Data Embedding"""

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma

# Load Arabic sentence embedding model (no trust_remote_code needed)
embedding_model_name = "omarelshehy/Arabic-Retrieval-v1.0"
embedding_model = SentenceTransformer(embedding_model_name)

# Define a custom embedding wrapper for ChromaDB
class ArabicEmbeddings:
    def embed_documents(self, texts):
        return embedding_model.encode(texts).tolist()  # Convert NumPy array to list

    def embed_query(self, text):
        return embedding_model.encode([text])[0].tolist()  # Single text embedding

# Initialize embeddings
embeddings = ArabicEmbeddings()

"""### 4.Data Storing"""

import os
import hashlib

# Define ChromaDB path
CHROMA_PATH = "vec_db"

# Function to generate a unique ID based on document content
def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()  # Hash-based unique ID

# Load existing database if it exists
if os.path.exists(CHROMA_PATH):
    db_chroma = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Fetch existing document texts and compute their IDs
    stored_docs = db_chroma.get(include=["documents"])["documents"]  # Retrieve stored texts
    existing_ids = {generate_id(doc) for doc in stored_docs}  # Compute existing document IDs
else:
    db_chroma = None
    existing_ids = set()

# Prepare new documents with unique IDs
new_texts = []  # List to store new document texts
new_metadatas = []  # List to store corresponding metadata
new_ids = []  # List to store unique document IDs

for chunk in chunks:
    chunk_text = chunk.page_content  # Get text content
    doc_id = generate_id(chunk_text)  # Generate unique ID

    if doc_id not in existing_ids:  # Avoid re-adding duplicates
        new_texts.append(chunk_text)
        new_metadatas.append(chunk.metadata)
        new_ids.append(doc_id)

# Add only unique documents
if new_texts:
    if db_chroma is None:
        # If DB was not initialized, create it with new documents
        db_chroma = Chroma.from_texts(new_texts, embeddings, metadatas=new_metadatas, ids=new_ids, persist_directory=CHROMA_PATH)
    else:
        # Correct method for adding new texts
        db_chroma.add_texts(new_texts, metadatas=new_metadatas, ids=new_ids)

# Persist database
if db_chroma:
    db_chroma.persist()

"""## Data Retrieval and Generation

### 1.Retrieval
"""

# this is an example of a user question (query)
query = 'طلب استرجاع هبة'

# retrieve context - top 50 most relevant (closests) chunks to the query vector
# (by default Langchain is using cosine distance metric)
docs_chroma = db_chroma.similarity_search_with_score(query, k=20)

# generate an answer based on given user query and retrieved context information
context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])


PROMPT_TEMPLATE = """
جاوب على السؤال بناءً على المحتوى التالي:
{context}

**سياق المحادثة السابقة:**
{chat_history}

**1- التحقق من المواد القانونية ذات الصلة:**
- ابحث في الملفات المتاحة عن أي مواد قانونية مرتبطة بالقضية المطروحة.
- اذكر رقم المادة ونصها كما هو مذكور في المصدر.
- حدد الملفات التي تحتوي على هذه المواد القانونية.

**2- فحص القضايا السابقة المشابهة:**
- استخرج القضايا السابقة التي تشابه القضية الحالية من حيث الوقائع أو الأحكام.
- قدم ملخصًا موجزًا عن كل قضية مشابهة، مع الإشارة إلى الفروقات أو التشابهات الجوهرية.
- ذكر رقم الصفحة والملف الذي يحتوي على هذه القضايا.

**3- استخراج النقاط المهمة التي قد تكون منسية في القضايا المشابهة:**
- قم بتحليل الأنماط المتكررة في القضايا المشابهة وحدد أي نقاط مهمة غالبًا ما يتم تجاهلها.
- قم بإبراز هذه النقاط وشرح مدى أهميتها في القضية الحالية.

**4- تقديم إجابة واضحة ومنظمة دون أي إشارة إلى تعديلات لغوية:**
- استخدم لغة دقيقة وسهلة الفهم دون الإشارة إلى أي تصحيحات أو تعديلات.
- في حال وجود تعارض بين الأرقام المكتوبة بالكلمات والأرقام الرقمية، اعتمد على النص المكتوب بالكلمات.
- أضف اسم الملف ورقم الصفحة أو المادة التي استندت إليها الإجابة لكل نقطة يتم ذكرها.
- قدم ملخصًا نهائيًا بسيطًا يوضح الإجابة بشكل مباشر وواضح.

جاوب على هذا السؤال: {question}

اعطِ إجابة مفصلة ومنظمة وفق الخطوات المذكورة أعلاه وصحح الأخطاء الإملائية بدون تغيير بالمحتوى بحيث يكون الرد سهل للقراءة.
كل الكلام يجب أن يكون باللغة العربية.
"""


# print(context_text)
# print(query)

"""### 2.Generation"""

from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()

# get GROQ API key from environment variable
groq_api = os.getenv("GROQ_API_KEY")

LLM_model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api)

# A function to be used by the backend of the website to generate the response to the user query
def generate_response_v2(query: str, chat_history: list, return_chat_name: bool=False) -> list:
    """
    Generate the response to the user query using the GROQ model.

    Args:
        query (str): The user query.
        chat_id (str): The unique identifier for the chat session.
        chat_history (list): The chat history for the chat session. (message_id: int, message_content: str, isBot:bool)
        return_chat_name (bool): Whether to return the name of the chat session.

    Returns:
        int: the status of the response (0 for success, 1 for failure).
        str: The generated response to the user query.
        str: The name of the chat session (if return_chat_name is True).
    """

    trimmed_messages = ""
    
    if len(chat_history) > 0:
        messages = []
        for message in chat_history:
            for _, message_content, isBot in message:
                if not isBot:
                    messages.append(HumanMessage(message_content))
                elif isBot:
                    messages.append(AIMessage(message_content))
        
        trimmed_messages = trim_messages(
            messages,
            strategy="last",
            token_counter=ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api),
            max_tokens=250,
            start_on="human",
            end_on=("human", "tool"),
            include_system=True,
            allow_partial=True
        )
        trimmed_messages = "\n".join([t.content for t in trimmed_messages])
    
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, chat_history=trimmed_messages, question=query)
    response_text = LLM_model.invoke(prompt)

    if return_chat_name:
        chat_name = LLM_model.invoke("Give me a sentence as a name for this chat if the first question is " + query + ". Return only the name and nothing else. Limit the name to 15 characters max. Make it readable and understandable.")
        return [0, response_text.content, chat_name.content]
    return [0, response_text.content, ""]

# Testing the function

# _, answer, name = generate_response(query, context_text, return_chat_name=True)
# print(answer)
# print("-"*50)
# print(name)

# # set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# # load retrieved context and user query in the prompt template
# prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
# prompt = prompt_template.format(context=context_text, question=query)

# # call LLM model to generate the answer based on the given context and query
# model = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api)
# response_text = model.invoke(prompt)

# print(response_text.content) # return response_text.content