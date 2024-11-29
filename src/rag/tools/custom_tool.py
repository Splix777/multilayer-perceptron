import os

from crewai_tools import GithubSearchTool
from crewai_tools import DirectoryReadTool
from crewai_tools import CodeDocsSearchTool


github_tool = GithubSearchTool(
	github_repo="https://github.com/Splix777/multilayer-perceptron",
	gh_token="ghp_......",
	content_types=["code", "repo", "pr"],
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3.2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="mxbai-embed-large",
                # task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)

directory_tool = DirectoryReadTool(
	directory=os.environ.get('PWD', os.getcwd())
	)

python_docs_tool = CodeDocsSearchTool(
	docs_url="https://docs.python.org/3.13/",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama3.2",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="ollama",
            config=dict(
                model="mxbai-embed-large",
                # task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)