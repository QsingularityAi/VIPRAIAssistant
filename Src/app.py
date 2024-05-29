from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,
)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler
from llms import get_multimodal_llm, get_graq_model

import chainlit as cl


embeddings_model = OpenAIEmbeddings()

CODE_STORAGE_PATH = "/Data"
#CODE_STORAGE_PATH = "/Users/anuragtrivedi/Desktop/VIRP_Project/NewData"
def process_python_files(py_storage_path: str):
    py_directory = Path(py_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for py_path in py_directory.glob("*.py"):
        loader = TextLoader(str(py_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_python_files"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search


# def process_python_files(pdf_storage_path: str):
#     pdf_directory = Path(pdf_storage_path)
#     docs = []  # type: List[Document]
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#     for pdf_path in pdf_directory.glob("*.pdf"):
#         loader = PyMuPDFLoader(str(pdf_path))
#         documents = loader.load()
#         docs += text_splitter.split_documents(documents)

#     doc_search = Chroma.from_documents(docs, embeddings_model)

#     namespace = "chromadb/my_documents"
#     record_manager = SQLRecordManager(
#         namespace, db_url="sqlite:///record_manager_cache.sql"
#     )
#     record_manager.create_schema()

#     index_result = index(
#         docs,
#         record_manager,
#         doc_search,
#         cleanup="incremental",
#         source_id_key="source",
#     )

#     print(f"Indexing stats: {index_result}")

#     return doc_search


doc_search = process_python_files(CODE_STORAGE_PATH)
model = get_graq_model()                                  #ChatOpenAI(model_name="gpt-4", streaming=True)


@cl.on_chat_start
async def on_chat_start():
    template = """Use the following piece of source information and provide a detailed summary of 
    its content, including descriptions of functions, classes, and any relevant calculations or algorithms. 
    If you don't know the answer, just say that you don't know, don't try to generate any random answer from your own

    Context:{context}
    Question:{question}

    Only return the helpful answer and nothing else
    helpful answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever(search_kwargs={'k': 5})

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata['source'], d.metadata['page'])
                self.sources.add(source_page_pair)  

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(
                    cl.Text(name="Sources", content=sources_text, display="inline")
                )

    async with cl.Step(type="run", name="QA Assistant"):
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[
                cl.LangchainCallbackHandler(),
                PostMessageHandler(msg)
            ]),
        ):
            await msg.stream_token(chunk)

    await msg.send()