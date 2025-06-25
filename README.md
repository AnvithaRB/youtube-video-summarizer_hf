# YouTube Video Summarizer using HuggingFace & LangChain

This project extracts the transcript from a YouTube video and summarizes it using a BART-based HuggingFace pipeline with LangChain.

# Overview

This Python script:
- Accepts a YouTube video URL or ID as input
- Fetches the video's transcript using the `youtube_transcript_api`
- Uses the BART summarization model (`facebook/bart-large-cnn`) to summarize the transcript
- Outputs the result as a clean, Markdown-formatted summary (`summary.md`)

# Example video used
[https://youtu.be/eBz7iUJu9UM?si=gr-4wCFNHpRiTfpI]

**Generated Summary Preview**:
```
# Key Points
In a 2019 study, over 400 participants were enlisted to learn a mysterious, invented language. In the first round, participants either had all their answers marked as correct no matter what, or they were forced to fail every question. The successful participants from round one rose to the top of the ranks, while those cast as failures kept, well, failing.

```

# Usage

# Clone this repository
```bash
git clone https://github.com/AnvithaRB/youtube-video-summarizer_hf
cd youtube-video-summarizer_hf
```

# Create a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: .\env\Scripts\activate
```

# Install dependencies
```bash
pip install -r requirements.txt
```

# Run the summarizer
```bash
python summarizer.py <YouTube-URL-or-ID>
```

Example:
```bash
python summarizer.py https://youtu.be/eBz7iUJu9UM?si=gr-4wCFNHpRiTfpI
```

The summary will be saved to a file called `summary.md`.

# Tech Stack

- Python
- [LangChain](https://www.langchain.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- BART model (`facebook/bart-large-cnn`)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)

# Files

- `summarizer.py` – Main script
- `summary.md` – Auto-generated summary output
- `requirements.txt` – All Python dependencies
- `README.md` – Project instructions