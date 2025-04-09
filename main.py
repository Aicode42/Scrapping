import asyncio
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from playwright.async_api import async_playwright
from dotenv import load_dotenv

# Set the event loop policy for Windows to support subprocesses
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Load environment variables
load_dotenv()

# Configure the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize FastAPI app
app = FastAPI()

# Define input model for /analyze endpoint
class AnalyzeInput(BaseModel):
    urls: str
    query: str

# Agent class for browser automation and analysis
class Agent:
    def __init__(self, task, llm, headless=True):
        self.task = task
        self.llm = llm
        self.headless = headless

    async def run(self):
        urls = self.task.split("Analyze these websites: ")[1].split(". ")[0].split(", ")
        query = self.task.split(". ")[1]

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            page = await browser.new_page()
            analysis = ""

            # Visit each URL and scrape basic content
            for url in urls:
                try:
                    await page.goto(url.strip(), timeout=60000)
                    content = await page.content()
                    # Simple extraction (could be enhanced with more logic)
                    analysis += f"Website: {url}\nContent Preview: {content[:200]}...\n\n"
                except Exception as e:
                    analysis += f"Website: {url}\nError: {str(e)}\n\n"

            # Use LLM to analyze based on query
            prompt = f"{self.task}\n\nScraped Data:\n{analysis}"
            response = await self.llm.apredict(prompt)
            await browser.close()
            return response

# Analyze endpoint
@app.post("/analyze")
async def analyze_websites(input_data: AnalyzeInput):
    try:
        urls = input_data.urls
        query = input_data.query

        if not urls or not query:
            return {"error": "Please provide both URLs and a query"}

        # Create the task string
        task = f"Analyze these websites: {urls}. {query}"

        # Initialize the Agent with headless mode
        agent = Agent(
            task=task,
            llm=llm,
            headless=True
        )

        # Run the agent asynchronously
        result = await agent.run()

        return {
            "status": "success",
            "task": task,
            "result": result
        }
    except Exception as e:
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
