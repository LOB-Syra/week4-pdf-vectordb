import os
from dotenv import load_dotenv
# 向量数据库
from langchain.vectorstores import Chroma
# 文档加载器
from langchain.document_loaders import PyPDFLoader
# 文本转换为向量的嵌入引擎
from langchain.embeddings.openai import OpenAIEmbeddings
# 文本拆分
from langchain.text_splitter import RecursiveCharacterTextSplitter



# 加载环境变量
load_dotenv(dotenv_path='.env')

# get a token: https://platform.openai.com/account/api-keys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

pdf_base_dir = "data"

doc = []
#遍历data文件夹下的文件
for item in os.listdir(pdf_base_dir):
    loader = PyPDFLoader(file_path=os.path.join(pdf_base_dir, item))
    #把每个文件都加入doc列表
    doc.append(loader.load())

#统计总共的文本量
print("提取文本量：", len(doc))
# 拆分
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
docs = []
for d in doc:
    docs.append(text_splitter.split_documents(d))
    print("拆分文档数：", len(docs))
# 准备嵌入引擎
embeddings = OpenAIEmbeddings()
# 向量化
# 会对 OpenAI 进行 API 调用
vectordb = Chroma(embedding_function=embeddings, persist_directory="./vectordb")
for d in docs:
    vectordb.add_documents(d)
# 持久化
vectordb.persist()