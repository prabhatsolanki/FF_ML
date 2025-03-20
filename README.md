Requirements:

      1. Python
      2. PyTorch
      3. Pyro
      4. Numpy, Seaborn, Matplotlib, Scipy, Scikit-learn

or just activate:

      source /afs/cern.ch/work/p/prsolank/public/flafenv/bin/activate
      
Usage:

       python fake_factor_v1p0.py --mode train --synthetic True --N 500000
       python fake_factor_v1p0.py --mode test --synthetic True
       
If ```--synthetic``` is ``False``, real data should be provided (currently returns None).
