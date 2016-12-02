#!/usr/bin/env python
#
# File: run_multiwalker.py
#
# Created: Friday, September  2 2016 by rejuvyesh <mail@rejuvyesh.com>
#
import numpy as np
from runners import RunnerParser

from madrl_environments.sniper import sniper
from madrl_environments.sniper.utils import TwoDMaps
from madrl_environments import StandardizedEnv, ObservationBuffer

# yapf: disable
ENV_OPTIONS = [
    ('n_targets', int, 2, ''),
    ('n_snipers', int, 2, ''),
    ('obs_range', int, 3, ''),
    ('map_size', str, '10,10', ''),
    ('map_type', str, 'rectangle', ''),
    ('n_catch', int, 2, ''),
    ('urgency', float, 0.0, ''),
    ('surround', int, 1, ''), #n_evaders must be >= 4 for surround to work 
    ('map_file', str, None, ''),
    ('sample_maps', int, 0, ''),
    ('flatten', int, 1, ''),
    ('reward_mech', str, 'local', ''),
    ('catchr', float, 0.1, ''),
    ('term_sniper', float, 5.0, ''),
    ('buffer_size', int, 1, ''),
]
# yapf: enable

def main(parser):
    mode = parser._mode
    args = parser.args

    if args.map_file:
        map_pool = np.load(args.map_file)
    else:
        if args.map_type == 'rectangle':
            #passes in tuple of what map should be 
            env_map = TwoDMaps.rectangle_map(*map(int, args.map_size.split(',')))
        elif args.map_type == 'complex':
            env_map = TwoDMaps.complex_map(*map(int, args.map_size.split(',')))
        else:
            raise NotImplementedError()
        #map pool is list of maps of different shapes for environment 
        map_pool = [env_map]


    env = sniper(map_pool, n_targets=args.n_targets, n_snipers=args.n_snipers,
                       obs_range=args.obs_range, n_catch=args.n_catch,
                       urgency_reward=args.urgency,
                       surround=bool(args.surround), sample_maps=bool(args.sample_maps),
                       flatten=bool(args.flatten),
                       reward_mech=args.reward_mech,
                       catchr=args.catchr,
                       term_sniper=args.term_sniper)

    if args.buffer_size > 1:
        env = ObservationBuffer(env, args.buffer_size)

    if mode == 'rllab':
        from runners.rurllab import RLLabRunner
        run = RLLabRunner(env, args)
    elif mode == 'rltools':
        from runners.rurltools import RLToolsRunner
        run = RLToolsRunner(env, args)
    else:
        raise NotImplementedError()

    run()

if __name__ == '__main__':
    main(RunnerParser(ENV_OPTIONS))
