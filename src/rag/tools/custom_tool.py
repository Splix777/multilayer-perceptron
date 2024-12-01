import os

from crewai_tools import (
    CSVSearchTool,
    DirectoryReadTool,
    FileReadTool,
    WebsiteSearchTool,
    SeleniumScrapingTool,
)


provider = "ollama"
llama_model = "llama3"
embedder_model = "mxbai-embed-large"
base_url = "http://ollama:11434"
custom_config = dict(
    llm=dict(
        provider=provider, config=dict(model=llama_model, base_url=base_url)
    ),
    embedder=dict(
        provider=provider, config=dict(model=embedder_model, base_url=base_url)
    ),
)

csv_search_tool: CSVSearchTool = CSVSearchTool(
    csv="data/csv/data.csv", config=custom_config
)

directory_read_tool: DirectoryReadTool = DirectoryReadTool(
    directory=os.path.join(os.getcwd(), "mlp"), config=custom_config
)

# We will need to get the directory from the directory_read_tool
file_read_tool: FileReadTool = FileReadTool(config=custom_config)

website_search_tool: SeleniumScrapingTool = SeleniumScrapingTool(
    website_url="https://en.wikipedia.org/wiki/Neural_network_(machine_learning)",
    config=custom_config,
)
