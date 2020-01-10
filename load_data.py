from collections import namedtuple
from pathlib import Path
from typing import List

Datapoint = namedtuple('Datapoint', ['x1', 'x2', 'x3', 'y', 'task'])


def load_data(path: Path = Path('./data/google-analogies.txt')) -> List[Datapoint]:
    dataset = []
    with path.open() as f:
        for line in f.readlines():
            if line[0] == ':':
                task = line[1:].strip(' \n')
                continue

            words = line.strip('\n').split(' ')
            p = Datapoint(*words, task)
            dataset.append(p)

    return dataset
