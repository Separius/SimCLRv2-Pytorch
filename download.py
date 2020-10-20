import os
import argparse
from math import ceil

import requests
from tqdm import tqdm

available_simclr_models = ['r50_1x_sk0', 'r50_1x_sk1', 'r50_2x_sk0', 'r50_2x_sk1',
                           'r101_1x_sk0', 'r101_1x_sk1', 'r101_2x_sk0', 'r101_2x_sk1',
                           'r152_1x_sk0', 'r152_1x_sk1', 'r152_2x_sk0', 'r152_2x_sk1', 'r152_3x_sk1']
simclr_base_url = 'https://storage.googleapis.com/simclr-checkpoints/simclrv2/{category}/{model}/'
files = ['checkpoint', 'graph.pbtxt', 'model.ckpt-{category}.data-00000-of-00001',
         'model.ckpt-{category}.index', 'model.ckpt-{category}.meta']
simclr_categories = {'finetuned_100pct': 37535, 'finetuned_10pct': 3754,
                     'finetuned_1pct': 751, 'pretrained': 250228, 'supervised': 28151}
chunk_size = 1024 * 8


def download(url, destination):
    if os.path.exists(destination):
        return
    response = requests.get(url, stream=True)
    save_response_content(response, destination)


def save_response_content(response, destination):
    if 'Content-length' in response.headers:
        total = int(ceil(int(response.headers['Content-length']) / chunk_size))
    else:
        total = None
    with open(destination, 'wb') as f:
        for data in tqdm(response.iter_content(chunk_size=chunk_size), leave=False, total=total):
            f.write(data)


def run():
    parser = argparse.ArgumentParser(description='Model Downloader')
    parser.add_argument('model', type=str, choices=available_simclr_models)
    parser.add_argument('--simclr_category', type=str, choices=list(simclr_categories.keys()), default='pretrained')
    args = parser.parse_args()
    model = args.model
    os.makedirs(model, exist_ok=True)
    url = simclr_base_url.format(model=model, category=args.simclr_category)
    model_category = simclr_categories[args.simclr_category]
    for file in tqdm(files):
        f = file.format(category=model_category)
        download(url + f, os.path.join(model, f))


if __name__ == '__main__':
    run()
