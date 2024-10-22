from langchain.chat_models.gigachat import GigaChat

from langchain.schema import HumanMessage, SystemMessage

from typing import List, Union

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding


# # 1. GigaChat
# Define GigaChat throw langchain.chat_models

def get_giga(giga_key: str) -> GigaChat:
    giga = GigaChat(credentials=giga_key, temperature=0.001,
                    model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)
    giga.verbose = False
    return giga


# # 2. Prompting
# ### 2.1 Define classic prompt

# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[Union[SystemMessage, HumanMessage]]:
    return [
        SystemMessage(
            content="You are a bot who helps the user to solve problems."),
        HumanMessage(content=user_content)
    ]


# ### 3. Define few-shot prompting


# Implement a function to build a few-shot prompt to count even digits in the
# given number. The answer should be in the format
# 'Answer: The number {number} consist of {text} even digits.', for example
# 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    content = f'''
        Even digits are 0, 2, 4, 6, 8, all other digits are not even",

        How many even digits number <number> consist of?

        In order to answer question you can delete all odd digits,
        and count the length of the remaining number.
        For example:
        How many even digits number 123456 consist of?
        1) Remove 1 3 5 because they are odd.
        2) Remaining part is 246 and the length of it is three.
        Answer: The number 123456 consist of three even digits.

        How many even digits number 2828 consist of?
        1) We do not remove anything, because all the digits are even.
        2) Remaining part is 2828 and the length of it is four.
        Answer: The number 2828 consist of four even digits.

        How many even digits number 7777 consist of?
        1) We remove 7777 because they are odd.
        2) Remaining part is `` and the length of it is zero.
        Answer: The number 7777 consist of zero even digits.

        How many even digits number 481356902779261 consist of?
        1) We remove 1 3 5 9 779 1 because they are odd.
        2) Remaining part is 4860226 and the length of it is seven.
        Answer: The number 481356902779261 consist of seven even digits.

        Here is some examples:
        Answer: The number 2 consist of one even digits.
        Answer: The number 9 consist of zero even digits.
        Answer: The number 4 consist of one even digits.
        Answer: The number 3 consist of zero even digits.
        Answer: The number 0 consist of one even digits.
        Answer: The number 82 consist of two even digits.
        Answer: The number 75 consist of zero even digits.
        Answer: The number 44 consist of two even digits.
        Answer: The number 31 consist of zero even digits.
        Answer: The number 76 consist of one even digits.
        Answer: The number 2468 consist of four even digits.
        Answer: The number 1357 consist of zero even digits.
        Answer: The number 123456 consist of three even digits.
        Answer: The number 98765432 consist of four even digits.
        Answer: The number 2288 consist of four even digits.
        Answer: The number 4466 consist of four even digits.

        How many even digits number {number} consist of?

        The answer should be in the format
        Answer: The number <number> consist of <count> even digits.
        Ensure that the answer don't have `s` at the end of `consist`.

        Evaluate the problem multiple times to ensure the anwer is consistent
        arcoss different evaluations.
    '''

    return [
        HumanMessage(content=content)
    ]


# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code 
# to build llama_index across your own documents. For this task you should use 
# GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt = """
        You are a Q&A assistant. Your goal is to answer questions as
        accurately as possible based on the instructions and context provided.
        """
        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        ))
        self.documents = SimpleDirectoryReader(path_to_data).load_data()
        self.service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=self.embed_model
        )
        self.index = VectorStoreIndex.from_documents(
            self.documents, service_context=self.service_context)

        self.query_engine = self.index.as_query_engine()

    def query(self, user_prompt: str) -> str:
        prompt = self.system_prompt + user_prompt
        response = self.query_engine.query(prompt)
        return response
