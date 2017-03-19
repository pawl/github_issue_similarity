import logging
import string

import click
import nltk
from gensim import corpora, models, similarities
from github import Github
from sqlalchemy import create_engine, Column, Integer, String, TEXT, ForeignKey
from sqlalchemy.orm import relationship, scoped_session, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

import settings

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
log_format = '[%(asctime)s] %(levelname)s - %(message)s'
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

github = Github(settings.github_token)

engine = create_engine(settings.db_uri, convert_unicode=True)
session = scoped_session(sessionmaker(autocommit=False,
                                      autoflush=False,
                                      bind=engine))
Base = declarative_base()
Base.query = session.query_property()


class Repo(Base):
    __tablename__ = 'repos'
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    full_name = Column(String(255))

    issues = relationship('Issue', lazy='joined')


class Issue(Base):
    __tablename__ = 'issues'
    id = Column(Integer, primary_key=True)
    title = Column(String(255))
    body = Column(TEXT(collation='utf8mb4_unicode_ci'), nullable=True)
    html_url = Column(String(255))
    state = Column(String(255))
    repo_id = Column(Integer, ForeignKey('repos.id'))


Base.metadata.create_all(engine)


@click.group()
def cli():
    pass


@cli.command()
def drop_tables():
    logger.info('Dropping database tables...')
    Base.metadata.drop_all(engine)
    logger.info('Finished dropping database tables...')


@cli.command()
@click.argument('repo_name')
def get_issues(repo_name):
    logger.info('Finding repo: %s', repo_name)
    repo = github.get_repo(repo_name)
    session.merge(Repo(
        id=repo.id,
        name=repo.name,
        full_name=repo.full_name,
    ))

    logger.info('Getting issues...')
    for issue in repo.get_issues():
        logger.info(issue)
        session.merge(Issue(
            id=issue.id,
            title=issue.title,
            body=issue.body,
            html_url=issue.html_url,
            state=issue.state,
            repo_id=repo.id,
        ))
    session.commit()

    logger.info('Finished getting issues.')


stop_words = nltk.corpus.stopwords.words('english')


def lemmatize(text):
    all_words = []

    split_text = text.split()
    for word in split_text:
        stripped_word = word.strip(string.punctuation)
        if stripped_word and (stripped_word not in stop_words):
            lemma = nltk.WordNetLemmatizer().lemmatize(word.lower())
            all_words.append(lemma)

    return all_words


def remove_single_occurences(texts):
    """remove words that occur only once to reduce the size"""
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens)
                      if all_tokens.count(word) == 1)
    return [[word for word in text if word not in tokens_once]
            for text in texts]


@cli.command()
@click.argument('repo_name')
def parse_issues(repo_name):
    logger.info('Querying for repo: %s', repo_name)
    repo = Repo.query.filter_by(full_name=repo_name).first()
    if not repo:
        logger.error('Unable to find repo. Try using the get_issues command.')
        return
    logger.info('Found repo.')

    logger.info('Tokenizing text, removing junk, and single occurences...')
    issues = [issue for issue in repo.issues]
    texts = [lemmatize(issue.body) for issue in repo.issues]
    texts = remove_single_occurences(texts)

    logger.info('Building a model...')
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    logger.info('Initializing the tfidf model...')
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]

    logger.info('Initializing the lsi model...')
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=15)
    lsi.print_topics(5)

    for text in texts:
        vec_bow = dictionary.doc2bow(text)
        vec_lsi = lsi[vec_bow]  # convert the query to LSI space

        logger.info('Transforming corpus to LSI space and index it IN RAM!')
        index = similarities.MatrixSimilarity(lsi[corpus])

        # perform a similarity query against the corpus and sort them
        sims = index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])

        # print out nicely the first 10 films
        for i, (document_num, sim) in enumerate(sims): # print sorted (document number, similarity score) 2-tuples
            if sim > 0.98:
                print issues[document_num].url, str(sim)


if __name__ == '__main__':
    cli()
