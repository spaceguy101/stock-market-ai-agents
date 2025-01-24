import requests
from typing import Type
from crewai.tools import tool, BaseTool
from bs4 import BeautifulSoup
import logging
from pydantic import BaseModel, Field
#from langchain_google_genai import (ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,HarmBlockThreshold,
#    HarmCategory,)
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
import logging
from crewai import LLM , Agent, Task
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

llm = LLM(model="ollama/llama3.2:latest")

class ScrapeAndSummariseWebsiteInput(BaseModel):
    source_url: str = Field(..., description="Source URL to be scraped", examples=["https://www.bbc.com/news/business-60500000"])

class ScrapeAndSummariseWebsite(BaseTool):
    name: str = "Scrape website content"
    description: str = """Useful to scrape and summarize a website content with financial news, blog, etc."""
    args_schema: Type[BaseModel] = ScrapeAndSummariseWebsiteInput

    def _run(self, source_url:str) -> str:
        response = requests.get(source_url)
        logger.info(f"Fetching content from {source_url}")
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            relevant_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'ul', 'ol', 'li', 'div', 'section', 'article'])    
            content = "\n\n".join([str(el) for el in relevant_elements])
            content_chunks = [content[i:i + 8000] for i in range(0, len(content), 8000)]
            summaries = []
            for chunk in content_chunks:
                agent = Agent(
                    role='Principal Researcher',
                    goal=
                    'Do amazing research and summaries based on the content you are working with',
                    backstory=
                    "You're a Principal Researcher at a big company and you need to do research about a given topic.",
                    allow_delegation=False,
                    llm=llm)
                task = Task(
                    agent=agent,
                    description=
                    f'Analyze and summarize the content below, make sure to include the most relevant information in the summary, return only the summary nothing else.\n\nCONTENT\n----------\n{chunk}',
                    expected_output='A paragraph of the summary of the content provided. Should be detailed.'
                )
                summary = task.execute()
                summaries.append(summary)
            logger.info(f"Successfully fetched content from {source_url}")
            logger.info(f"Summaries: {summaries}")
            return "\n\n".join(summaries)
        else:
            logger.error(f"Failed to fetch content from {source_url}. Status code: {response.status_code}")
            return f"Failed to fetch content from {source_url}. Status code: {response.status_code}"