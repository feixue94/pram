import argparse
import torch
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint

import localization.matchers as matchers
from localization.base_model import dynamic_load
from colmap_utils.parsers import names_to_pair

confs = {
    'gm': {
        'output': 'gm',
        'model': {
            'name': 'gm',
            'weight_path': '/scratches/flyer_3/fx221/exp/uniloc/20230519_145436_gm_L9_resnet4x_B16_K1024_M0.2_relu_bn_adam/gm.900.pth',
            'sinkhorn_iterations': 20,
        },
    },
    'gml': {
        'output': 'gml',
        'model': {
            'name': 'gml',
            'weight_path': '/scratches/flyer_3/fx221/exp/uniloc/20240117_145500_gml2_L9_resnet4x_B32_K1024_M0.2_relu_bn_adam/gml2.920.pth',
            'sinkhorn_iterations': 20,
        },
    },

    'adagml': {
        'output': 'adagml',
        'model': {
            'name': 'adagml',
            'weight_path': '/scratches/flyer_3/fx221/exp/uniloc/20240210_154552_adagml2_L9_resnet4x_B32_K1024_M0.2_relu_bn_adam/adagml2.80.pth',
            'sinkhorn_iterations': 20,
        },
    },

    'superglue': {
        'output': 'superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'weight_path': '/scratches/flyer_2/fx221/Research/Code/third_weights/superglue_outdoor.pth',
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


@torch.no_grad()
def main(conf, pairs, features, export_dir, exhaustive=False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    feature_path = Path(export_dir, features + '.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')
    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    elif exhaustive:
        logging.info(f'Writing exhaustive match pairs to {pairs}.')
        assert not pairs.exists(), pairs

        # get the list of images from the feature file
        images = []
        feature_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = list(set(images))

        pair_list = [' '.join((images[i], images[j]))
                     for i in range(len(images)) for j in range(i)]
        with open(str(pairs), 'w') as f:
            f.write('\n'.join(pair_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_name = f'{features}-{conf["output"]}-{pairs_name}'
    match_path = Path(export_dir, match_name + '.h5')

    # if os.path.exists(match_path):
    #     logging.info('Matching file exists.')
    #     return match_path
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            # data[k + '0'] = feats0[k].__array__()
            data[k + '0'] = feats0[k][()]
        for k in feats1.keys():
            # data[k + '1'] = feats1[k].__array__()
            data[k + '1'] = feats1[k][()]
        data = {k: torch.from_numpy(v)[None].float().to(device)
                for k, v in data.items()}

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])

        pred = model(data)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')

    return match_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, required=True, choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir,
         exhaustive=args.exhaustive)
