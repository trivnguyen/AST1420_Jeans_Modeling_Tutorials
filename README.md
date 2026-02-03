# Jeans Modeling Tutorials

This repo contains the tutorials for Jeans modeling of spherical systems for Project I of AST1420: Galactic Dynamics.

To run the tutorials, you will need to have Python installed along with the following packages:
- numpy
- scipy
- matplotlib
- emcee
- corner
- astropy
- pocomc
- multiprocess

You can install these packages using pip:

```bash
pip install numpy scipy matplotlib emcee corner astropy pocomc multiprocess
```

I recommend using a virtual environment to manage your Python packages. You can create a virtual environment using `venv`:

```bash
python -m venv jeans_env
source jeans_env/bin/activate
```

There are two notebooks (including two solution notebooks):
- `01_constant_beta.ipynb`: This notebook covers Jeans modeling with a constant anisotropy parameter.
- `02_OM_beta.ipynb`: This notebook covers Jeans modeling with an Osipkov-Merritt anisotropy profile.

You should work through the notebooks in order, starting with `01_constant_beta.ipynb`.
