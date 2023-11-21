import os
from time import perf_counter
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

DATA_FOLDER = './SOURCE_DOCUMENTS'

def pdf_loader(data_folder=DATA_FOLDER):
    print([fn for fn in os.listdir(DATA_FOLDER) if fn.endswith('.pdf')])
    loaders = [PyPDFLoader(os.path.join(DATA_FOLDER, fn))
               for fn in os.listdir(DATA_FOLDER) if fn.endswith('.pdf')]
    print(f'{len(loaders)} file loaded')
    return loaders

def build_qa_chain(chunk_size: int = 1000, chunk_overlap: int = 50) -> RetrievalQA:
    embedding = OpenAIEmbeddings()
    splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Replace 'gpt-4-model-version' with the specific version of GPT-4 you wish to use, e.g., 'gpt-4-0613'
    llm = OpenAI(model_name="gpt-4",
                 temperature=0.9,
                 max_tokens=256)  # Check the appropriate value for max_tokens for GPT-4

    loaders = pdf_loader()
    index = VectorstoreIndexCreator(
        embedding=embedding,
        text_splitter=splitter).from_loaders(loaders)
    print(len(index.vectorstore.get()))

    # Prepare the pipeline
    return RetrievalQA.from_chain_type(llm=llm,
                                       chain_type="stuff",
                                       retriever=index.vectorstore.as_retriever(search_type="similarity",
                                                                                search_kwargs={"k": 4}),
                                       return_source_documents=True,
                                       input_key="question")

tick = perf_counter()
qa_chain = build_qa_chain(chunk_overlap=0)
print(f'Time span for building index: {perf_counter() - tick}')

# get reply to our questions
tick = perf_counter()
result = qa_chain({'question': 'co wiesz o ART deals?', 'include_run_info': True})

print(f'Time span for query: {perf_counter() - tick}')

print('Q:', result['question'])
print('A:', result['result'])
print('\n')
print('Resources:', result['source_documents'])