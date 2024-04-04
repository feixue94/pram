import argparse
import torch
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import localization.matchers as matchers
from localization.base_model import dynamic_load
from colmap_utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval

confs = {
    'gm': {
        'output': 'gm',
        'model': {
            'name': 'gm',
            'weight_path': 'weights/imp_gm.900.pth',
            'sinkhorn_iterations': 20,
        },
    },
    'gml': {
        'output': 'gml',
        'model': {
            'name': 'gml',
            'weight_path': 'weights/imp_gml.920.pth',
            'sinkhorn_iterations': 20,
        },
    },

    'adagml': {
        'output': 'adagml',
        'model': {
            'name': 'adagml',
            'weight_path': 'weights/imp_adagml.80.pth',
            'sinkhorn_iterations': 20,
        },
    },

    'superglue': {
        'output': 'superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'weight_path': 'weights/superglue_outdoor.pth',
        },
    },
    'NNM': {
        'output': 'NNM',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },
}


class WorkQueue:
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,)) for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_q, feature_path_r):
        self.pairs = pairs
        self.feature_path_q = feature_path_q
        self.feature_path_r = feature_path_r

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        data = {}
        with h5py.File(self.feature_path_q, "r") as fd:
            grp = fd[name0]
            for k, v in grp.items():
                data[k + "0"] = torch.from_numpy(v.__array__()).float()
                if k == 'descriptors':
                    data[k + '0'] = data[k + '0'].t()
            # some matchers might expect an image but only use its size
            data["image0"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        with h5py.File(self.feature_path_r, "r") as fd:
            grp = fd[name1]
            for k, v in grp.items():
                data[k + "1"] = torch.from_numpy(v.__array__()).float()
                if k == 'descriptors':
                    data[k + '1'] = data[k + '1'].t()
            data["image1"] = torch.empty((1,) + tuple(grp["image_size"])[::-1])
        return data

    def __len__(self):
        return len(self.pairs)


def writer_fn(inp, match_path):
    pair, pred = inp
    with h5py.File(str(match_path), "a", libver="latest") as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred["matches0"][0].cpu().short().numpy()
        grp.create_dataset("matches0", data=matches)
        if "matching_scores0" in pred:
            scores = pred["matching_scores0"][0].cpu().half().numpy()
            grp.create_dataset("matching_scores0", data=scores)


def main(
        conf: Dict,
        pairs: Path,
        features: Union[Path, str],
        export_dir: Optional[Path] = None,
        matches: Optional[Path] = None,
        features_ref: Optional[Path] = None,
        overwrite: bool = False,
) -> Path:
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError(
                "Either provide both features and matches as Path" " or both as names."
            )
    else:
        if export_dir is None:
            raise ValueError(
                "Provide an export_dir if features is not" f" a file path: {features}."
            )
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}-{conf["output"]}-{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                        names_to_pair(i, j) in fd
                        or names_to_pair(j, i) in fd
                        or names_to_pair_old(i, j) in fd
                        or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
        conf: Dict,
        pairs_path: Path,
        match_path: Path,
        feature_path_q: Path,
        feature_path_ref: Path,
        overwrite: bool = False,
) -> Path:
    logging.info(
        "Matching local features with configuration:" f"\n{pprint.pformat(conf)}"
    )

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logging.info("Skipping the matching.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(matchers, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=1, shuffle=False, pin_memory=True
    )
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        data = {
            k: v if k.startswith("image") else v.to(device, non_blocking=True)
            for k, v in data.items()
        }
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))
    writer_queue.join()
    logging.info("Finished exporting matches.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, required=True, choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
