from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from datetime import date

RAG_GENERATOR = """
    Instructions: Give a reply to the query using the json chunk given as context.

    Always use latest data (timestamp is provided as milliseconds prefix to the financial metric).
    For example, if query is: 'what is the trailing PEG ratio?', you should look for 'trailingPegRatio' in the context
    or '[most_recent_timestamp_in_milliseconds]_trailingPegRatio' in the context.
    If you are asked a question that you know the answer of, feel free to use your own knowledge instead of the context provided.
    For example,if query is 'ABFRL|what is the company's sector?' and the answer is not in context, you can use your own knowledge about the company to answer this. 
    Make sure you only do this when you do not find any answer in the json chunk.
    ------------------
    You can match the user's query to these keys in the json.
    Possible keys: 
    [
        "Tax_Effect_Of_Unusual_Items",
        "Tax_Rate_For_Calcs",
        "Normalized_EBITDA",
        "Net_Income_From_Continuing_Operation_Net_Minority_Interest",
        "Reconciled_Depreciation",
        "Reconciled_Cost_Of_Revenue",
        "EBITDA",
        "EBIT",
        "Net_Interest_Income",
        "Interest_Expense",
        "Normalized_Income",
        "Net_Income_From_Continuing_And_Discontinued_Operation",
        "Total_Expenses",
        "Rent_Expense_Supplemental",
        "Diluted_Average_Shares",
        "Basic_Average_Shares",
        "Diluted_EPS",
        "Basic_EPS",
        "Diluted_NI_Availto_Com_Stockholders",
        "Net_Income_Common_Stockholders",
        "Otherunder_Preferred_Stock_Dividend",
        "Net_Income",
        "Minority_Interests",
        "Net_Income_Including_Noncontrolling_Interests",
        "Net_Income_Continuous_Operations",
        "Tax_Provision",
        "Pretax_Income",
        "Other_Non_Operating_Income_Expenses",
        "Net_Non_Operating_Interest_Income_Expense",
        "Interest_Expense_Non_Operating",
        "Operating_Income",
        "Operating_Expense",
        "Other_Operating_Expenses",
        "Depreciation_And_Amortization_In_Income_Statement",
        "Depreciation_Income_Statement",
        "Rent_And_Landing_Fees",
        "Gross_Profit",
        "Cost_Of_Revenue",
        "Total_Revenue",
        "Operating_Revenue"
    ]

    or

    [
        'address1', 'address2', 'city', 'zip', 'country', 'phone', 'website', 
        'industry', 'industryKey', 'industryDisp', 'sector', 'sectorKey', 'sectorDisp', 
        'longBusinessSummary', 'fullTimeEmployees', 'companyOfficers', 'maxAge', 
        'priceHint', 'previousClose', 'open', 'dayLow', 'dayHigh', 
        'regularMarketPreviousClose', 'regularMarketOpen', 'regularMarketDayLow', 'regularMarketDayHigh', 
        'beta', 'forwardPE', 'volume', 'regularMarketVolume', 'averageVolume', 
        'averageVolume10days', 'averageDailyVolume10Day', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 
        'priceToSalesTrailing12Months', 'fiftyDayAverage', 'twoHundredDayAverage', 'currency', 
        'enterpriseValue', 'profitMargins', 'floatShares', 'sharesOutstanding', 
        'heldPercentInsiders', 'heldPercentInstitutions', 'impliedSharesOutstanding', 'bookValue', 
        'priceToBook', 'lastFiscalYearEnd', 'nextFiscalYearEnd', 'mostRecentQuarter', 
        'netIncomeToCommon', 'trailingEps', 'forwardEps', 'pegRatio', 
        'enterpriseToRevenue', 'enterpriseToEbitda', '52WeekChange', 'SandP52WeekChange', 
        'exchange', 'quoteType', 'symbol', 'underlyingSymbol', 
        'shortName', 'longName', 'firstTradeDateEpochUtc', 'timeZoneFullName', 
        'timeZoneShortName', 'uuid', 'messageBoardId', 'gmtOffSetMilliseconds', 
        'currentPrice', 'targetHighPrice', 'targetLowPrice', 'targetMeanPrice', 
        'targetMedianPrice', 'recommendationMean', 'recommendationKey', 'numberOfAnalystOpinions', 
        'totalCash', 'totalCashPerShare', 'ebitda', 'totalDebt', 
        'totalRevenue', 'debtToEquity', 'revenuePerShare', 'revenueGrowth', 
        'grossMargins', 'ebitdaMargins', 'operatingMargins', 'financialCurrency', 
        'trailingPegRatio'
    ]
    ------------------
    Context: 
    {context}
    ------------------
    Query: {question}
    Answer: 
"""

def predict(question, vector_index, prompt):
    llm = ChatOpenAI(
        model="gpt-40-mini",
        temperature=0.1,
        max_retries=2,
        api_key=os.environ.get("OPENAI_API_KEY")
    )


    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)
    qa_chain = (
        {
            "context": vector_index.as_retriever(),
            "question": RunnablePassthrough(),
        }
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(question)
    return result["result"]

def loading_content():
    jq_schema = '.'
    chunks = []

    json_loader = JSONLoader(file_path='financial_data.json', jq_schema=jq_schema, text_content=False)

    single_post_chunks = json_loader.load_and_split()

    chunks.extend(single_post_chunks)

    return chunks


def chunking():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=150)
    context = f"Today's Date: {date.today()}\n\n".join(str(p.page_content) for p in loading_content())
    texts = text_splitter.split_text(context)
    return texts, context


def init_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    vector_index = Chroma.from_texts(texts, embeddings).as_retriever(search_kwargs={"k":10})

    return vector_index


def search(user_request):
    texts, context = chunking()
    vector_index = init_vectorstore(texts)
    answer = predict(user_request, vector_index, RAG_GENERATOR)
    return answer