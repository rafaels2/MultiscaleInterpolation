# MultiscalteInterpolation
## How to run?
```bash
pip instasll -r requirements.txt
python interpolation.py
```
## Modules 
- `interpolation.py` is the main module. It runs the multiscale logic on a specific method.
- `quasi_interpolation.py` now includes an environemnt averaging using RBF coefficients.
- `naive.py` picks the RBFs' coefficients by solving interpolation constaint.
- `utils.py` is as it sounds :)
- `config.py` configures the run.
