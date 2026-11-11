# Contact-state inversion of pressed Cu powder

Data and the main steps of the MD runs and of the network solution for "Contact-State Inversion Constrains Heat Transport in Gas-Solid Powder Thermal Conductors".

- data: measurements, DEM parameters, contact networks of the packing cores, MD conductances with standard errors, results with their uncertainty
- structures: representative interface structures
- md: main steps of the MD runs (the first lines of each input say what it reads; settings in Table A1 of the paper)
- code: network solution

    python code/network_model.py

The script solves the contact networks for kc and k/kc. The source column of data/results_uncertainty.csv tells which values the script gives and which come from the analysis in the paper.
