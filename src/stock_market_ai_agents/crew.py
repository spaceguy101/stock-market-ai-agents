import logging
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from stock_market_ai_agents.tools.browser_tools import ScrapeAndSummariseWebsite
from stock_market_ai_agents.tools.search_tools import search_internet,search_news
from stock_market_ai_agents.tools.calculator_tools import CaclulatorTool
from stock_market_ai_agents.tools.finance_tools import search_annual_income_statement,  search_stock_fundamentals, SearchQuaterlyIncomeStatement
from crewai import LLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

llm = LLM(model="ollama/llama3.2:latest")

search_quarterly_income_statement = SearchQuaterlyIncomeStatement()
calculate = CaclulatorTool()
scrape_and_summarize_website = ScrapeAndSummariseWebsite()

@CrewBase
class StockMarketAiAgents():
	"""StockMarketAiAgents crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def financial_analyst(self) -> Agent:
		logger.info("Initializing Financial Analyst Agent")
		return Agent(
			config=self.agents_config['financial_analyst'],
			verbose=True,
			tools=[
				scrape_and_summarize_website,
				search_internet,
				calculate,
				search_annual_income_statement,
				search_quarterly_income_statement,
				search_stock_fundamentals
			],
			llm=llm,
         	memory=True
		)

	@agent
	def research_analyst(self) -> Agent:
		logger.info("Initializing Research Analyst Agent")
		return Agent(
			config=self.agents_config['research_analyst'],
			verbose=True,
			tools=[
				scrape_and_summarize_website,
				search_internet,
				calculate,
				search_annual_income_statement,
				search_quarterly_income_statement,
				search_stock_fundamentals
			],
			llm=llm,
			memory=True
		)
  
	@agent
	def investment_advisor(self) -> Agent:
		logger.info("Initializing Investment Advisor Agent")
		return Agent(
			config=self.agents_config['investment_advisor'],
			verbose=True,
			tools=[
				scrape_and_summarize_website,
				search_internet,
				search_news,
				calculate
			],
			llm=llm,
         	memory=True
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def stock_analysis_task(self) -> Task:
		logger.info("Initializing Stock Analysis Task")
		return Task(
			config=self.tasks_config['stock_analysis_task'],
		)

	@task
	def financial_analysis_task(self) -> Task:
		logger.info("Initializing Financial Analysis Task")
		return Task(
			config=self.tasks_config['financial_analysis_task'],
		)
  
	@task
	def filings_analysis_task(self) -> Task:
		logger.info("Initializing Filings Analysis Task")
		return Task(
			config=self.tasks_config['filings_analysis_task'],
		)
  
	@task
	def recommendation_task(self) -> Task:
		logger.info("Initializing Recommendation Task")
		return Task(
			config=self.tasks_config['recommendation_task'],
		)

	@crew
	def crew(self) -> Crew:
		logger.info("Creating the StockMarketAiAgents crew")
		"""Creates the StockMarketAiAgents crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
