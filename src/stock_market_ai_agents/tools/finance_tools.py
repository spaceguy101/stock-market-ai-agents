from typing import Type
from crewai.tools import tool,BaseTool
import yfinance as yf
import json
from datetime import datetime
from ..utils import search
import logging
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

income_stmt_keys = [
"Tax Effect Of Unusual Items",
"Tax Rate For Calcs",
"Normalized EBITDA",
"Net Income From Continuing Operation Net Minority Interest",
"Reconciled Depreciation",
"Reconciled Cost Of Revenue",
"EBITDA",
"EBIT",
"Net Interest Income",
"Interest Expense",
"Normalized Income",
"Net Income From Continuing And Discontinued Operation",
"Total Expenses",
"Rent Expense Supplemental",
"Diluted Average Shares",
"Basic Average Shares",
"Diluted EPS",
"Basic EPS",
"Diluted NI Availto Com Stockholders",
"Net Income Common Stockholders",
"Otherunder Preferred Stock Dividend",
"Net Income",
"Minority Interests",
"Net Income Including Noncontrolling Interests",
"Net Income Continuous Operations",
"Tax Provision",
"Pretax Income",
"Other Non Operating Income Expenses",
"Net Non Operating Interest Income Expense",
"Interest Expense Non Operating",
"Operating Income",
"Operating Expense",
"Other Operating Expenses",
"Depreciation And Amortization In Income Statement",
"Depreciation Income Statement",
"Rent And Landing Fees",
"Gross Profit",
"Cost Of Revenue",
"Total Revenue",
"Operating Revenue"
]

class FinanceTools:
    @staticmethod
    def __store_financial_data(data, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        logger.info(f"Storing financial data to {file_path}")
        print("JSON data has been stored in", file_path)

    @staticmethod
    def __embedding_search(data, ask):
        logger.info(f"Performing embedding search for query: {ask}")
        FinanceTools.__store_financial_data(data, 'financial_data.json')
        return search(ask)

    @staticmethod
    def preprocess(json_data):
        global income_stmt_keys

        new_data = {}

        for key, value in json_data.items():
            new_dict = {}
            for stmt_key in income_stmt_keys:
                new_dict[key + "_" + stmt_key.replace(" ", "_")] = value.get(stmt_key, None)
            new_data.update(new_dict)

        new_json = json.dumps(new_data, indent=4)
        logger.info("Preprocessing financial data")
        return new_json


class SearchQuaterlyIncomeStatementInput(BaseModel):
    company: str = Field(..., description="The company's stock symbol",examples=["RELIANCE"])
    query: str = Field(..., description="The question you have about the company's quarterly income statement",examples=["What is the net income?"])

class SearchQuaterlyIncomeStatement(BaseTool):
    name: str = "Search quarterly income statement"
    description: str = """
    Useful to search information for a given stock.

    Available data for past 4 quarters: [
    "Tax Effect Of Unusual Items",
    "Tax Rate For Calcs",
    "Normalized EBITDA",
    "Net Income From Continuing Operation Net Minority Interest",
    "Reconciled Depreciation",
    "Reconciled Cost Of Revenue",
    "EBITDA",
    "EBIT",
    "Net Interest Income",
    "Interest Expense",
    "Normalized Income",
    "Net Income From Continuing And Discontinued Operation",
    "Total Expenses",
    "Rent Expense Supplemental",
    "Diluted Average Shares",
    "Basic Average Shares",
    "Diluted EPS",
    "Basic EPS",
    "Diluted NI Availto Com Stockholders",
    "Net Income Common Stockholders",
    "Otherunder Preferred Stock Dividend",
    "Net Income",
    "Minority Interests",
    "Net Income Including Noncontrolling Interests",
    "Net Income Continuous Operations",
    "Tax Provision",
    "Pretax Income",
    "Other Non Operating Income Expenses",
    "Net Non Operating Interest Income Expense",
    "Interest Expense Non Operating",
    "Operating Income",
    "Operating Expense",
    "Other Operating Expenses",
    "Depreciation And Amortization In Income Statement",
    "Depreciation Income Statement",
    "Rent And Landing Fees",
    "Gross Profit",
    "Cost Of Revenue",
    "Total Revenue",
    "Operating Revenue"
    ]
    """
    args_schema: Type[BaseModel] = SearchQuaterlyIncomeStatementInput

    def _run(self, company:str, query:str) -> str:
        global income_stmt_keys
        stock = company
        ask = query
        stock_data = yf.Ticker(f"{stock}.NS") # only NSE stocks
        data = FinanceTools.preprocess(json.loads(stock_data.quarterly_income_stmt.to_json()))
        logger.info(f"Searching quarterly income statement for query: {query}")
        answer = FinanceTools.__embedding_search(data, ask)
        return answer

    
@tool("Search stock fundamentals")
def search_stock_fundamentals(company:str,query: str) -> str:
    """
    Useful to search information for a given stock.

    Available data for the stock: [
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
    """
    global income_stmt_keys
    stock = company
    ask = query
    stock_data = yf.Ticker(f"{stock}.NS") # only NSE stocks
    data = stock_data.info
    logger.info(f"Searching stock fundamentals for query: {query}")
    answer = FinanceTools.__embedding_search(data, ask)
    return answer
    
@tool("Search annual income statement")
def search_annual_income_statement(company:str,query:str) -> str:
    """
    Useful to search information for a given stock.
    The input to this tool should be a symbol and what question you have from it.

    Available data for past 4 years: [
    "Tax Effect Of Unusual Items",
    "Tax Rate For Calcs",
    "Normalized EBITDA",
    "Net Income From Continuing Operation Net Minority Interest",
    "Reconciled Depreciation",
    "Reconciled Cost Of Revenue",
    "EBITDA",
    "EBIT",
    "Net Interest Income",
    "Interest Expense",
    "Normalized Income",
    "Net Income From Continuing And Discontinued Operation",
    "Total Expenses",
    "Rent Expense Supplemental",
    "Diluted Average Shares",
    "Basic Average Shares",
    "Diluted EPS",
    "Basic EPS",
    "Diluted NI Availto Com Stockholders",
    "Net Income Common Stockholders",
    "Otherunder Preferred Stock Dividend",
    "Net Income",
    "Minority Interests",
    "Net Income Including Noncontrolling Interests",
    "Net Income Continuous Operations",
    "Tax Provision",
    "Pretax Income",
    "Other Non Operating Income Expenses",
    "Net Non Operating Interest Income Expense",
    "Interest Expense Non Operating",
    "Operating Income",
    "Operating Expense",
    "Other Operating Expenses",
    "Depreciation And Amortization In Income Statement",
    "Depreciation Income Statement",
    "Rent And Landing Fees",
    "Gross Profit",
    "Cost Of Revenue",
    "Total Revenue",
    "Operating Revenue"
    ]
    """
    global income_stmt_keys
    stock = company
    ask = query
    stock_data = yf.Ticker(f"{stock}.NS") # only NSE stocks
    data = FinanceTools.preprocess(json.loads(stock_data.income_stmt.to_json()))
    logger.info(f"Searching annual income statement for query: {query}")
    answer = FinanceTools.__embedding_search(data, ask)
    return answer
