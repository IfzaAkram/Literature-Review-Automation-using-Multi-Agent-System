# Literature Review using Multi-Agent System

This notebook demonstrates the process of writing a literature review using a multi-agent system. The agents will download the papers from the provided links, read and summarize each paper, and then generate the literature review from these summaries.

**Step 1:** Install and configure all the necessary packages.

!pip install PyPDF2 pdfplumber pytesseract pdf2image Rouge textstat scikit-learn arxiv
!pip install autogen autogen_ext autogen_agentchat
!pip install --upgrade opentelemetry-sdk

import requests
from getpass import getpass
import PyPDF2
import asyncio
import os
from autogen import UserProxyAgent, AssistantAgent, GroupChat, GroupChatManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from getpass import getpass
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


**Step 2:** Setup Groq API
import os
from getpass import getpass

tokenGROQ = getpass('Enter GROQ_API_KEY here: ')
os.environ["GROQ_API_KEY"] = tokenGROQ
print(os.environ.get("GROQ_API_KEY"))






# Create the model client
model_client = OpenAIChatCompletionClient(
    model="gemma2-9b-it",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unkown",
    },
)


