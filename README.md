# week4-pdf-vectordb
- 使用langchain中的pypdfloader对pdf文件进行读取
- 使用RecursiveCharacterTextSplitter对文本进行拆分
- 使用OpenAIEmbeddings将文本转换为向量的嵌入引擎
- 调用langchain的Chroma向量数据库，将文本进行存储，存储到本地新建"./vectordb"文件夹，运行后文件夹生成chroma.sqlite3文件
