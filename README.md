# Test case at Troople 

## Features
- Create 2 flask app
	- First Flask application contains buttons to start and stop sreen recording
	- Second Flask application receives images (screen) and add them process them to add them to the Chromadb. This second flask app should also be used as API to answer to questions about the screenshots.

## Installation

setup .env file with appropriate variables

```bash
pip install -r requirements.txt
```

## Usage

run the main.py in interface_chromedb_src and then run interface.py in another terminal.

