# app.py

from flask import Flask, jsonify, request
from celery import Celery
from crawler import Crawler
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

crawler = Crawler()

@celery.task
def crawl_url(url):
    result = crawler.crawl(url)
    return {'url': url, 'result': result}

@app.route('/crawl', methods=['POST'])
def start_crawl():
    data = request.get_json()
    url = data['url']
    task = crawl_url.delay(url)
    logging.debug(f'Starting crawl for url: {url}')
    return jsonify({'url': url, 'task_id': task.id}), 202

@app.route('/crawl/status/<task_id>', methods=['GET'])
def get_crawl_status(task_id):
    task = crawl_url.AsyncResult(task_id)
    response = {'status': task.status}
    if task.status == 'SUCCESS':
        response['result'] = task.get()
    logging.debug(f'Status: {response}')
    return jsonify(response), 202

@app.route('/search', methods=['GET'])
def search():
    data = request.get_json()
    query = data['query']
    threshold = data['threshold']
    logging.debug(f'Search for url: {query}')
    app.logger.info('Received query: %s', query)
    results = crawler.search(query,threshold)
    return jsonify({'result': results}), 202

@app.route('/wn_search', methods=['GET'])
def wn_search():
    data = request.get_json()
    query = data['query']
    threshold = data['threshold']
    logging.debug(f'Search wordnet for url: {query}')
    app.logger.info('Received query: %s', query)
    results = crawler.wn_search(query,threshold)
    return jsonify({'result': results}), 202

if __name__ == '__main__':
    app.run(debug=True)