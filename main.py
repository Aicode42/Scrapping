import asyncio
import os
import subprocess
import sys
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller, BrowserConfig, Browser
from dotenv import load_dotenv

# Initialize FastAPI app
app = FastAPI()

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Can be DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Log to a file named app.log
        logging.StreamHandler()         # Optional: Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Set the event loop policy for Windows to support subprocesses
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Check and install Playwright if not already installed
def ensure_playwright_installed():
    playwright_installed_flag = os.path.join(os.path.expanduser("~"), ".playwright_installed")
    if not os.path.exists(playwright_installed_flag):
        try:
            print("Installing Playwright dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"])
            subprocess.check_call(["playwright", "install"])
            # Create a flag file to indicate Playwright is installed
            with open(playwright_installed_flag, "w") as f:
                f.write("Playwright installed")
            print("Playwright installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install Playwright: {e}")
            sys.exit(1)
    else:
        print("Playwright is already installed, skipping installation.")

# Run the Playwright installation check at startup
ensure_playwright_installed()

# Load environment variables


# Define the output format as a Pydantic model
class Post(BaseModel):
    text: str


class Posts(BaseModel):
    posts: List[Post]


# Define input model for /analyze endpoint
class AnalyzeInput(BaseModel):
    urls: str
    query: str

# Analyze endpoint
@app.post("/analyze")
async def analyze_websites(input_data: AnalyzeInput):
    try:
        urls = input_data.urls
        query = input_data.query

        if not urls or not query:
            return {"error": "Please provide field"}

        # Create the task string
        task = f"{query} for this : {urls}"

        # Configure LLM
        llm = ChatOpenAI(
            model="gpt-4o",  # Or your preferred model
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Initialize the Controller
        controller = Controller(output_model=Posts)
        config = BrowserConfig(headless=True, disable_security=True)
        browser = Browser(config=config)

        # Initialize the Agent
        agent = Agent(
            task=task,
            llm=llm,
            controller=controller,
            browser=browser,
        )

        # Run the agent asynchronously
        history = await agent.run()

        # Extract the result and parse to a list of `Post` objects
        result = history.final_result()
        if result:
            parsed_posts: Posts = Posts.model_validate_json(result)
            result_text = parsed_posts.model_dump_json(indent=2)  # Format as JSON
        else:
            result_text = "No results found."

        # Log the processed result
        logger.info(f"Processed result sent to frontend: {result_text}")

        return {
            "status": "success",
            "task": task,
            "result": result_text
        }

    except Exception as e:
        logger.error(f"Error in analyze_websites: {str(e)}")
        return {"error": str(e)}

# Root endpoint serving HTML
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Website Analyzer</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            textarea, input {
                width: 100%;
                margin-bottom: 10px;
                padding: 8px;
            }
            button {
                padding: 10px 20px;
                background-color: #4CAF50;
                color: white;
                border: none;
                cursor: pointer;
            }
            #result {
                margin-top: 20px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h1>Website Analyzer</h1>
        <div>
            <label for="urls">Enter URLs or Keywords (comma-separated):</label>
            <textarea id="urls" rows="3" placeholder="(https://example1.com, https://example2.com) (Industry Trends, future technologies)"></textarea>
            
            <label for="query">Enter Query:</label>
            <input type="text" id="query" placeholder="Content strategy for Q1 product launches and promotions">
            
            <button onclick="analyze()">Analyze</button>
            
            <div id="result"></div>
        </div>

        <script>
            async function analyze() {
                const urls = document.getElementById('urls').value;
                const query = document.getElementById('query').value;
                const resultDiv = document.getElementById('result');
                
                resultDiv.innerHTML = 'Analyzing...';

                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            urls: urls,
                            query: query
                        })
                    });

                    const data = await response.json();
                    
                    if (data.error) {
                        resultDiv.innerHTML = `Error: ${data.error}`;
                    } else {
                        resultDiv.innerHTML = `Task: ${data.task}\n\nResult:\n${data.result}`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `Error: ${error.message}`;
                }
            }
        </script>
    </body>
    </html>
    """

# Run the app with Uvicorn
if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Get the port from Render
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
