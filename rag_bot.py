import time
from operator import itemgetter
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langsmith import traceable
from nltk.tokenize import word_tokenize


class RagBot:

    def __init__(self, retriever, model, is_local_model):
        self._retriever = retriever
        # Wrapping the client instruments the LLM
        self._model = model
        self._is_local_model = is_local_model
        self._prompt = self.prompt_template()

    # Set up the prompt template
    def prompt_template(self):
        return ChatPromptTemplate.from_messages(
            [
                ("system",
                 """You are a helpful assistant that can answer questions about technical documents in any language. 
                 Keep your answers only in the language of the question(s).

                 Only use the factual information from the document(s) to answer the question(s). Keep your answers concise and to the point.

                 If you do not have have sufficient information to answer a question, politely refuse to answer and say "I don't know".
                 \n\nRelevant documents will be retrieved below."""
                 "Context: {context}"
                 ),
                ("human", "{question}"),
            ])

    @traceable()
    def retrieve_docs(self, question):
        return self._retriever.invoke(question)

    @traceable()
    def invoke_llm(self, query, docs):
        chain = (
            # {"docs": retriever,"question": RunnablePassthrough()}
                {"context": itemgetter("context"), "question": itemgetter("question")}
                | self._prompt | self._model | StrOutputParser()
        )

        # Visualize input schema if needed
        # chain.input_schema.schema()
        # Retrieve context docs
        # context = retriever.invoke(query)

        print(f"Question : \n{query}\n\n")
        # Stream the result if HuggingFaceEndpoint is used
        result = ""
        stopwatch = time.perf_counter()  # measure time
        if not self._is_local_model:
            print(f"Invoking the result with Inference API...\n")
            chunks = []
            result = chain.invoke({"question": query, "context": docs})
            print(result)
            # for chunk in chain.stream({"context": context, "question": query}):
            #     result+=chunk
            #     print(chunk, end='|', flush=True)

        else:
            print(f"Invoking the result with Local LLM...\n")
            result = chain.invoke({"context": docs, "question": query})
            # result.append(chunk)
            # print(chunk, end='|', flush=True)
        print(f"\n\nTime for invoke {(time.perf_counter() - stopwatch) / 60}")
        print(f"\nThe answer is based on the following {self._retriever.k} relevant documents:")
        # context = result.get("context", []) # Retrieve the context
        for doc in docs:
            print(f"\n{doc.page_content}\nMetadata: {doc.metadata}\n")

        # Evaluators will expect "answer" and "contexts"
        return {
            "answer": result,
            "contexts": docs,
        }

    @traceable()
    def get_answer(self, query: str):
        docs = self.retrieve_docs(query)
        return self.invoke_llm(query, docs)

