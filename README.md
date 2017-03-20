# Installation
1. Start MySQL
1. Install the requirements with `pip install -r requirements.txt`.
1. Run `python -c 'import nltk; nltk.download()'` and install wordnet, stopwords, and comtrans from under the "Corpora" tab.

Note: You may need to update setuptools to prevent errors while installing gensim.

# Usage
1. Fill in your github token and MySQL database uri in settings.py.
1. `python main.py get_issues '<full name of your repo>'`
1. `python main.py parse_issues '<full name of your repo>'`

Example of a full repo name: `twbs/bootstrap`
