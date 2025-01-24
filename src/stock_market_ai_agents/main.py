#!/usr/bin/env python
import sys
import warnings
import logging
from datetime import date
from stock_market_ai_agents.crew import StockMarketAiAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    logger.info("Running the crew")
    """
    Run the crew.
    """
    inputs = {
        'company': 'INFOLLION',
        'date': str(date.today())
    }
    
    try:
        StockMarketAiAgents().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    logger.info("Training the crew")
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "company": "INFOLLION "
    }
    try:
        StockMarketAiAgents().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    logger.info("Replaying the crew execution")
    """
    Replay the crew execution from a specific task.
    """
    try:
        StockMarketAiAgents().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    logger.info("Testing the crew execution")
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "company": "INFOLLION"
    }
    try:
        StockMarketAiAgents().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
