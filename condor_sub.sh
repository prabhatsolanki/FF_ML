#!/bin/bash
source /afs/cern.ch/work/p/prsolank/public/flafenv/bin/activate
python fake_factor_v5.py --mode train --synthetic True --N 1000000
