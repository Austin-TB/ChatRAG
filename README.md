# UVJ-Bot

## Usage
- Install the requirements into a an environment (conda, venv, uv)
- Activate the environment
- Populate your Gemini API key in the `.env` file
- For starting with the MCP server:
    - Uncomment MCP client implementation from `agent.py`  #12 -> #21
    - Run the MCP server in a seperate terminal:
      ```
      python mcp_server.py
      ```
- On initial run, we need to load the vector store if we haven't done it yet:
    - Run
      ```
      python vector_store.py
      ```
- Run the agent in Gradio or LangGraph UI
    - For gradio, run
      ```
      python app.py
      ```
    - for LangGraph UI, run
      ```
      langgraph dev
      ````
