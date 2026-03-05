# IndoLepAtlas Web Scraper

This repository contains a Python-based web scraper designed to compile a comprehensive dataset of butterflies, moths, and host plants from [ifoundbutterflies.org](https://www.ifoundbutterflies.org/).

The scraper extracts high-resolution images, taxonomic data, and metadata descriptions (like early stages, distribution, and larval host plants). To ensure minimal storage overhead on the host machine, it utilizes a **micro-batching** strategy: it downloads the images for a single species, generates the structured `metadata.jsonl`, syncs the batch directly to a Hugging Face Dataset repository, and immediately wipes the local temporary folder.

## Setup

1. **Environment Initialization:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Hugging Face Configuration:**
   Create a `.env` file in the root directory (this is omitted from version control via `.gitignore`) and insert your Hugging Face API token with **Write** access:
   ```text
   HF_TOKEN=your_huggingface_token_here
   ```

   *By default, the script pushes to the `AXE8/IndoLepAtlas` dataset. You can change `REPO_ID` inside `crawler.py` if needed.*

## Usage

To start the scraping process, simply run the crawler:

```bash
python3 crawler.py
```

### Running on a Remote/Shared Node

If you are running this on a shared node (like a college GPU server), it is highly recommended to run the script inside a detached `screen` or `tmux` session, so you can safely close your SSH connection without killing the process:

```bash
# Start a new screen session
screen -S scraper

# Activate the venv and run
source .venv/bin/activate
python3 crawler.py

# Detach from the session (Ctrl+A, then D)
```

## Architecture

* `crawler.py`: Scans the website's index to locate every species taxonomy page. It handles rate limiting, coordinates the Hugging Face `HfApi` upload, and manages local batch deletions.
* `scraper_prototype.py`: The core metadata extraction logic. It uses BeautifulSoup to parse taxonomy breadcrumbs, dynamically extract secondary metadata from HTML tabs, and scrape all 'Adult' and 'Early Stage' images into structured JSON schema.
