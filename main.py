# main.py

import os
from dotenv import load_dotenv
from data_retrieval import (
    get_company_news,
    get_stock_data,
    get_financial_statements,
    prepare_all_documents
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

def load_embedder():
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    return embedder

def build_vector_store(documents, embedder):
    vector_store = Chroma.from_documents(documents, embedder)
    return vector_store

def load_language_model():
    llm = OllamaLLM(model="llama2")
    return llm

def create_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever()

    prompt_template = """
You are a highly knowledgeable and professional financial analyst assistant with expertise in various aspects of finance, including but not limited to financial markets, stock analysis, economic indicators, company financial statements, and market trends.

**Your Objectives:**

1. **Provide Clear and Concise Answers**: Offer answers that are straightforward, well-structured, and easy to understand. Avoid unnecessary jargon. If technical terms are used, provide brief explanations to ensure the user fully comprehends the information.

2. **Strictly Use Provided Context**: Your responses must be **strictly based on the information provided in the "Context" section** below. Do not introduce any information that is not included in the context. If the answer is not found within the context, politely inform the user that the information is not available.

3. **Maintain Professionalism**: Use a neutral and professional tone at all times. Do not include personal opinions, biases, or emotions. Ensure that your language is respectful and appropriate for all users.

4. **Avoid Personalized Investment Advice**: Do not provide personalized investment recommendations or advice. If a user asks for such advice, gently remind them that you are not authorized to provide personalized investment guidance and that all information is for informational purposes only.

5. **Informational Purposes Only**: Any predictions, analyses, or forward-looking statements should be presented as informational and not as definitive forecasts. Use phrases like "Based on the provided information..." or "The context suggests that...".

6. **Handle Unavailable Information Appropriately**: If the necessary information to answer the user's question is not present in the context, respond by:

   - Politely informing the user that the information is not available.
   - Encouraging the user to provide additional context or data if possible.
   - Avoiding speculation or assumptions beyond the provided context.

7. **Ethical and Compliance Standards**: Adhere to all ethical guidelines and compliance standards, including confidentiality and data protection. Do not disclose any sensitive information.

8. **Formatting Guidelines**:

   - **Introduction**: Begin with a brief acknowledgment of the user's query.
   - **Main Content**: Present information in organized paragraphs or bullet points.
   - **Conclusion**: End with an offer for further assistance if appropriate.
   - **Clarity**: Use headings or subheadings if the answer is long or covers multiple topics.
   - **Numerical Data**: When citing figures or statistics from the context, ensure accuracy and clarity.

9. **Examples and Analogies**: If it helps clarify the information, use examples or simple analogies, but only if they are supported by the context.

10. **Language and Tone**:

    - Use formal language appropriate for professional communication.
    - Be polite and courteous.
    - Avoid colloquialisms, slang, or overly casual expressions.

Context:
{context}

Question:
{question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": PROMPT,
        },
        input_key="question"
    )

    print(f"Expected input keys: {qa_chain.input_keys}")

    return qa_chain

def main():
    load_dotenv()

    company_name = input("Enter the company name: ")
    ticker_symbol = input("Enter the company's stock ticker symbol: ")

    print("\nFetching data...")
    news_articles = get_company_news(company_name)
    stock_data = get_stock_data(ticker_symbol)
    financial_statements = get_financial_statements(ticker_symbol)

    # Verify data retrieval
    print("\nStock Data Retrieved:")
    print(stock_data.tail())

    print("\nFinancial Statements Retrieved:")
    for key, df in financial_statements.items():
        print(f"\n{key.capitalize()}:")
        print(df.head())

    print("\nPreparing documents for retrieval...")
    documents = prepare_all_documents(news_articles, financial_statements)

    # Include stock data if available
    if not stock_data.empty:
        stock_info = stock_data.tail(7).to_string()
        stock_doc = Document(
            page_content=f"Recent stock data:\n{stock_info}",
            metadata={"source": "stock_data"}
        )
        documents.append(stock_doc)
    else:
        print("No recent stock data available.")

    # Print documents for debugging
    for doc in documents:
        print(f"\nDocument content:\n{doc.page_content}\n")

    if not documents:
        print("No documents available to build the vector store. Exiting.")
        return

    print("\nBuilding vector store...")
    embedder = load_embedder()
    vector_store = build_vector_store(documents, embedder)

    print("\nLoading language model...")
    llm = load_language_model()

    print("\nSetting up QA chain...")
    qa_chain = create_qa_chain(llm, vector_store)

    print("\nChatbot is ready! Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() in ['exit', 'quit']:
            print("Chatbot: Goodbye!")
            break

        response = qa_chain.invoke({'question': user_query})
        print(f"Chatbot: {response.get('result') or response.get('answer')}")

if __name__ == '__main__':
    main()
