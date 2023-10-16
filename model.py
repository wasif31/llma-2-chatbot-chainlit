from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
DB_FAISS_PATH = "vectorstores/db_faiss"
custom_prompt_template = """USe the following pieces of information to answer the user's question.
If you dont know the answer , please just say that you dont know the answer, dont try to makeup an answer.
Context : {context}
Question :{question}
only return the answer bellow and nothing else.
Helpful answer: 

"""
def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vector stores
    """
    print(948)
    prompt = PromptTemplate(template = custom_prompt_template, input_variables = [
        'context','question'])
    print(949)
    return prompt

def load_llm():
    print(734)
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {'k':2}),
        return_source_documents = True,
        chain_type_kwargs= {'prompt':prompt}
    )
    return qa_chain
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-miniLM-L6-v2',
                                       model_kwargs = {'device': 'cpu'})
    print(443)
    db = FAISS.load_local(DB_FAISS_PATH,embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    print(10023)
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})
    return response
@cl.on_chat_start
async def start():
    print(22)
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi Welcome to Wasif's Bot. Why Are you here?"
    await msg.update()
    cl.user_session.set("chain",chain)

@cl.on_message
async def main(message):
    print(11)
    chain = cl.user_session.get("chain")
    print(12)
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True,
        answer_prefix_tokens= ["FINAL","ANSWER"]
    )
    print(13)
    cb.answer_reached = True
    res = await chain.acall(message,callbacks=[cb])
    print(14)
    answer = res["result"]
    print(15)
    sources = res["source_documents"]
    print(16)

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += f"\n No Sources Found"
    await cl.Message(content=answer).send()