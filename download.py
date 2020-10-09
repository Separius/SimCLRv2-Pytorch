import os
import argparse
from math import ceil

import requests
from tqdm import tqdm

available_models = ['r50_1x_sk0', 'r50_1x_sk1', 'r50_2x_sk0', 'r50_2x_sk1',
                    'r101_1x_sk0', 'r101_1x_sk1', 'r101_2x_sk0', 'r101_2x_sk1',
                    'r152_1x_sk0', 'r152_1x_sk1', 'r152_2x_sk0', 'r152_2x_sk1', 'r152_3x_sk1']
base_url = 'https://storage.googleapis.com/simclr-checkpoints/simclrv2/pretrained/{model}/'
files = ['checkpoint', 'graph.pbtxt', 'model.ckpt-250228.data-00000-of-00001',
         'model.ckpt-250228.index', 'model.ckpt-250228.meta']
chunk_size = 1024 * 8


def run():
    parser = argparse.ArgumentParser(description='SimCLR downloader')
    parser.add_argument('model', type=str, choices=available_models)
    model = parser.parse_args().model
    os.makedirs(model, exist_ok=True)
    url = base_url.format(model=model)
    for file in tqdm(files):
        destination = os.path.join(model, file)
        if os.path.exists(destination):
            continue
        response = requests.get(url + file, stream=True)
        with open(destination, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=chunk_size), leave=False,
                             total=int(ceil(int(response.headers['Content-length']) / chunk_size))):
                f.write(data)


if __name__ == '__main__':
    run()
