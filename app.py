import os
from operator import itemgetter
from pathlib import Path
from typing import List

import chainlit as cl
import chainlit.data as cl_data
from chainlit.types import ThreadDict
from langchain.callbacks.base import BaseCallbackHandler
from langchain.indexes import SQLRecordManager, index
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from feedback import CustomDataLayer

chunk_size = 1024
chunk_overlap = 50

embeddings_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    model="sentence-transformers/all-MiniLM-L12-v2",
)

# Feedback
cl_data._data_layer = CustomDataLayer()

PDF_STORAGE_PATH = "./data"


def process_pdfs(pdf_storage_path: str):
    pdf_directory = Path(pdf_storage_path)
    docs = []  # type: List[Document]
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    recursive_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    docs = recursive_text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("No documents found in the specified directory.")

    doc_search = Chroma.from_documents(docs, embeddings_model)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="full",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search


doc_search = process_pdfs(PDF_STORAGE_PATH)
# model = ChatOpenAI(model_name="gpt-4", streaming=True)
model = ChatGroq(
    model='llama-3.1-70b-versatile',
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=5,
    api_key=os.getenv("GROQ_API_KEY"),
    # other params...
)


@cl.on_chat_start
async def on_chat_start():

def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             """You are a helpful assistant that can answer questions about technical documents in any language. 
             Keep your answers only in the language of the question(s).

             Only use the factual information from the document(s) to answer the question(s). Keep your answers concise and to the point.

             If you do not have have sufficient information to answer a question, politely refuse to answer and say "I don't know".
             \n\nRelevant documents will be retrieved below."""
             "Context: {context}"
             ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever(search_kwargs={"k": 5})

    runnable = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RunnablePassthrough.assign(
                history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
            )
            | prompt
            | model
            | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()

@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    runnable = cl.user_session.get("runnable")  # type: Runnable
    response = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = []  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for doc in documents:
                source = doc.metadata.get('source', 'Unknown Source')
                page = doc.metadata.get('page', 'N/A')
                page_content = doc.page_content
                # self.sources.add(source_page_pair)  # Add unique pairs to the set
                if not any(s["source"] == source and s["page"] == page for s in self.sources):
                    self.sources.append({
                        "source": source,
                        "page": page,
                        "content": page_content
                    })

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                # Create a list of clickable elements for sources
                text_elements = []
                source_references = []
                for idx, src in enumerate(self.sources):
                    source_name = f"{src['source']} p.{src['page']}"
                    source_references.append(source_name)

                    # Add a previewable Chainlit element
                    text_elements.append(
                        cl.Text(
                            name=source_name,
                            content=src["content"],
                            display="side",
                        )
                    )
                # Generate the answer with clickable source names
                self.msg.content += f"\n\nSources: {", ".join(
                    source_references
                )}"

                # Append text elements to the message
                self.msg.elements.extend(text_elements)

    async for chunk in runnable.astream(
            message.content,
            config={
                "configurable": {"session_id": "foo"},
                "callbacks": [
                    cl.LangchainCallbackHandler(),
                    PostMessageHandler(response)
                ]
            },
    ):
        await response.stream_token(chunk)

    await response.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(response.content)