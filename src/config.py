import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
 
load_dotenv()
 
DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5442/postgres?gssencmode=disable",
)
 
TRIM_THRESHOLD = 20   # How many messages must accumulate to trigger summarization?
KEEP_RECENT    = 4    # Number of recent messages to retain after summarization
MAX_TOKENS     = 3000 # Trim safety threshold (approximate word count)
 
#Model
model = ChatAnthropic(model="claude-haiku-4-5")