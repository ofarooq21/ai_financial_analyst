# data_retrieval.py

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()
FMP_API_KEY = os.getenv('FMP_API_KEY')
SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY')

def get_company_news(company_name, num_results=5):
    """
    Fetches recent news articles related to the specified company using SerpAPI.
    """
    params = {
        'engine': 'google',
        'q': company_name,
        'tbm': 'nws',
        'num': num_results,
        'api_key': SERPAPI_API_KEY
    }
    try:
        response = requests.get('https://serpapi.com/search', params=params)
        response.raise_for_status()
        data = response.json()
        if 'error' in data:
            print(f"SERPAPI error: {data['error']}")
            return []
        else:
            return data.get('news_results', [])
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        return []

def get_stock_data(ticker_symbol, period='1mo', interval='1d'):
    """
    Retrieves historical stock data for the specified ticker symbol using yfinance.
    """
    import yfinance as yf
    import pandas as pd

    try:
        stock = yf.Ticker(ticker_symbol.upper())
        # Fetch data without specifying end date
        df = stock.history(period=period, interval=interval)
        if df.empty:
            print("No stock data found.")
            return pd.DataFrame()
        else:
            # Reset index to make 'Date' a column
            df.reset_index(inplace=True)
            # Ensure 'Date' is in datetime format
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            # Drop rows with NaT in 'Date' column
            df.dropna(subset=['Date'], inplace=True)
            # Get the last available date in the data
            last_available_date = df['Date'].max()
            # Filter data up to the last available date
            df = df[df['Date'] <= last_available_date]
            # Print sample data
            print("\nSample Stock Data:")
            print(df.head())
            return df
    except Exception as e:
        print(f"Exception while fetching stock data: {e}")
        return pd.DataFrame()

def get_financial_statements(ticker_symbol):
    """
    Fetches financial statements (income statement, balance sheet, cash flow) from the FMP API.
    """
    financial_statements = {}
    base_url = 'https://financialmodelingprep.com/api/v3'
    endpoints = {
        'income_statement': f'{base_url}/income-statement/{ticker_symbol.upper()}?apikey={FMP_API_KEY}&limit=5',
        'balance_sheet': f'{base_url}/balance-sheet-statement/{ticker_symbol.upper()}?apikey={FMP_API_KEY}&limit=5',
        'cash_flow': f'{base_url}/cash-flow-statement/{ticker_symbol.upper()}?apikey={FMP_API_KEY}&limit=5'
    }
    for statement_type, url in endpoints.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            # Print raw data for debugging
            print(f"\nRaw data for {statement_type}:\n{data}\n")
            if data:
                df = pd.DataFrame(data)
                # Convert 'date' column to datetime
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                # Drop rows with NaT in 'date' column
                df.dropna(subset=['date'], inplace=True)
                # Get the last available date in the data
                last_available_date = df['date'].max()
                # Filter data up to the last available date
                df = df[df['date'] <= last_available_date]
                if df.empty:
                    print(f"No valid data found for {statement_type}")
                    financial_statements[statement_type] = pd.DataFrame()
                else:
                    financial_statements[statement_type] = df
            else:
                print(f"No data found for {statement_type}")
                financial_statements[statement_type] = pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"Request exception for {statement_type}: {e}")
            financial_statements[statement_type] = pd.DataFrame()
    return financial_statements

def prepare_news_documents(news_articles):
    """
    Converts the list of news articles into Document objects for processing.
    """
    documents = []
    for article in news_articles:
        content = f"Title: {article.get('title')}\nSnippet: {article.get('snippet')}\nDate: {article.get('date')}\nLink: {article.get('link')}\n"
        metadata = {
            "source": "news",
            "title": article.get('title'),
            "date": article.get('date'),
            "link": article.get('link')
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def prepare_financial_documents(financial_statements):
    """
    Transforms the financial statements into Document objects for processing.
    """
    documents = []
    for statement_type, df in financial_statements.items():
        if not df.empty:
            for _, row in df.iterrows():
                date = row.get('date', 'N/A').strftime('%Y-%m-%d')
                content = f"{statement_type.replace('_', ' ').capitalize()} Report for {date}:\n"
                # Select key financial metrics to include
                if 'revenue' in row:
                    key_metrics = row[['revenue', 'netIncome', 'eps']]
                elif 'totalAssets' in row:
                    key_metrics = row[['totalAssets', 'totalLiabilities', 'totalStockholdersEquity']]
                elif 'operatingCashFlow' in row:
                    key_metrics = row[['operatingCashFlow', 'capitalExpenditure', 'freeCashFlow']]
                else:
                    key_metrics = row
                for col, value in key_metrics.items():
                    if col != 'date':
                        content += f"{col}: {value}\n"
                metadata = {
                    "source": statement_type,
                    "date": date
                }
                documents.append(Document(page_content=content, metadata=metadata))
        else:
            print(f"No data available to prepare documents for {statement_type}")
    return documents

def prepare_all_documents(news_articles, financial_statements):
    """
    Combines news and financial documents into a single list of Document objects.
    """
    news_docs = prepare_news_documents(news_articles)
    financial_docs = prepare_financial_documents(financial_statements)
    all_documents = news_docs + financial_docs
    return all_documents