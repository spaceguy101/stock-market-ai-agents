import json
import os
from crewai.tools import tool
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool("Search the internet")
def search_internet(query: str) -> str:
  """Useful to search the internet 
  about a a given topic and return relevant results"""
  top_result_to_return = 4
  url = "https://google.serper.dev/search"
  payload = json.dumps({"q": query})
  headers = {
      'X-API-KEY': os.environ['SERPER_API_KEY'],
      'content-type': 'application/json'
  }
  logger.info(f"Searching the internet for query: {query}")
  response = requests.request("POST", url, headers=headers, data=payload)
  logger.info(f"Received response: {response.status_code}")
  results = response.json()['organic']
  string = []
  for result in results[:top_result_to_return]:
    try:
      string.append('\n'.join([
          f"Title: {result['title']}", f"Link: {result['link']}",
          f"Snippet: {result['snippet']}", "\n-----------------"
      ]))
    except KeyError:
      next

  return '\n'.join(string)

@tool("Search news on the internet")
def search_news(query: str) -> str:
  """Useful to search news about a company, stock or any other
  topic and return relevant results"""""
  top_result_to_return = 4
  url = "https://google.serper.dev/news"
  payload = json.dumps({"q": query})
  headers = {
      'X-API-KEY': os.environ['SERPER_API_KEY'],
      'content-type': 'application/json'
  }
  logger.info(f"Searching news for query: {query}")
  response = requests.request("POST", url, headers=headers, data=payload)
  logger.info(f"Received response: {response.status_code}")
  results = response.json()['news']
  string = []
  for result in results[:top_result_to_return]:
    try:
      string.append('\n'.join([
          f"Title: {result['title']}", f"Link: {result['link']}",
          f"Snippet: {result['snippet']}", "\n-----------------"
      ]))
    except KeyError:
      next

  return '\n'.join(string)

