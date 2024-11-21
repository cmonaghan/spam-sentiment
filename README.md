# Spam sentiment analysis

This trains a masked language modeling (MLM) that can detect spam messages.

## Setup

We must use python 3.12 as `torch` is not yet supported on 3.13. You can use pyenv to install 3.12.

    python3.12 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Adding new requirements

    pip freeze > requirements.txt
