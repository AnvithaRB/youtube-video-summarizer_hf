import sys
import re
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langgraph.graph import StateGraph
from typing import TypedDict
from transformers import pipeline


def extract_video_id(url_or_id):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry["text"] for entry in transcript])



summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
llm = HuggingFacePipeline(pipeline=summarizer_pipeline)

prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are a summarizer agent. Summarize the following YouTube transcript in a clear and easy-to-understand way.
Start with a short overview, then list 4–6 main points as bullet points using Markdown format.

Transcript: {transcript}
"""
)
chain = LLMChain(llm=llm, prompt=prompt)


def start_node(state):
    return {"transcript": state["transcript"]}

def summarize_node(state):
    result = chain.invoke({"transcript": state["transcript"]})
    return {"summary": result["text"]}


class GraphState(TypedDict):
    transcript: str
    summary: str

builder = StateGraph(GraphState)

builder.add_node("start", start_node)
builder.add_node("summarize", summarize_node)
builder.set_entry_point("start")
builder.add_edge("start", "summarize")
builder.set_finish_point("summarize")

app = builder.compile()


def save_summary(summary_text):
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write("# Key Points\n\n")
        f.write(summary_text.strip())
    print("✅ Summary saved to summary.md")




def main():
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <YouTube URL or ID>")
        sys.exit(1)

    video_id = extract_video_id(sys.argv[1])
    print(f"📺 Fetching transcript for video ID: {video_id}")
    transcript = get_transcript(video_id)

    print("🧠 Summarizing with BART (local)...")
    state = {"transcript": transcript}
    result = app.invoke(state)
    save_summary(result["summary"])

if __name__ == "__main__":
    main()
