from typing import Type
from crewai.tools import tool, BaseTool
import logging
from pydantic import BaseModel, Field
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaclulatorToolInput(BaseModel):
    operation: str = Field(..., description="Mathematical operation to perform", examples=["200*7"])

class CaclulatorTool(BaseTool):
    name: str = "Make a calculation"
    description: str = """Useful to perform any mathematical calculations, 
    like sum, minus, multiplication, division, etc.
    The input to this tool should be a mathematical 
    expression, a couple examples are `200*7` or `5000/2*10`
    """
    args_schema: Type[BaseModel] = CaclulatorToolInput

    def _run(self, operation:str) -> str:
        logger.info(f"Performing calculation for operation: {operation}")
        return eval(operation)