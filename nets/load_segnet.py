# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   pram -> load_segnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/04/2024 15:39
=================================================='''
from nets.segnet import SegNet
from nets.segnetvit import SegNetViT


def load_segnet(network, n_class, desc_dim, n_layers, output_dim):
    model_config = {
        'network': {
            'descriptor_dim': desc_dim,
            'n_layers': n_layers,
            'n_class': n_class,
            'output_dim': output_dim,
            'with_score': False,
        }
    }

    if network == 'segnet':
        model = SegNet(model_config.get('network', {}))
        # config['with_cls'] = False
    elif network == 'segnetvit':
        model = SegNetViT(model_config.get('network', {}))
    else:
        raise 'ERROR! {:s} model does not exist'.format(config['network'])

    return model
