import pybamm
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import concurrent.futures
import os
import logging
import itertools
import glob
# logging.basicConfig(filename='simulation_errors.log', level=logging.ERROR)

pybamm.set_logging_level("NOTICE")

model_sei = pybamm.lithium_ion.DFN(
    {
        "SEI": ("reaction limited","none"),
        "SEI porosity change": "true",
    }
)

model_plating = pybamm.lithium_ion.DFN(
    {
        "SEI": ("reaction limited","none"),
        "SEI porosity change": "true",
        "lithium plating": ("partially reversible","none"),
        "lithium plating porosity change": "true",  
    }
)

param = pybamm.ParameterValues("Prada2013")
param["Current function [A]"] = 1.1
param["Nominal cell capacity [A.h]"] = 1.1
model_list = [model_sei, model_plating]
pm_list = ["SEI reaction exchange current density [A.m-2]", "Dead lithium decay constant [s-1]"]

def run_simulation(*pm_values):
    """
    Run the simulation based on the break points and models provided.
    """
    param_copy = param.copy()
    sol = None  # Reset solution for each run

    bat_id = pm_values[0]  
    sei_val = pm_values[1]
    plating_val = pm_values[2]
    crate = pm_values[3]
    break_points = pm_values[4]
    scale_factor = pm_values[5]
    param_copy[pm_list[0]] = sei_val
    param_copy[pm_list[1]] = plating_val
    param_copy["Initial concentration in negative electrode [mol.m-3]"] = 0.40 * 30555 * scale_factor
    param_copy["Initial concentration in positive electrode [mol.m-3]"] = 0.001 * 22806 * scale_factor

    pm_labels = [f"{sei_val:.3e}", f"{plating_val:.3e}", f"{crate:.3e}"]

    for i in range(len(break_points)):

        if sol is None:
            cycles = break_points[i]  
        else:
            cycles = break_points[i] - break_points[i-1]   
            filename = f"/home/chenhang/battery/250828/{bat_id}.pkl"
            filename = glob.glob(filename)
            sol = pickle.load(open(filename[0], "rb"))

        model = model_list[i]

        exp = pybamm.Experiment([
            (
                f"Charge at {crate}C until 3.2 V",
                f"Charge at 1C until 3.6 V",
                f"Hold at 3.6 V until C/50",
                f"Discharge at 4C until 2.0 V",
                f"Hold at 2.0 V until C/50",
            )
        ] * cycles,
        period="0.1 minute",
        temperature="30 oC")

        sim = pybamm.Simulation(
            model,
            experiment=exp,
            parameter_values=param_copy,
            solver=pybamm.CasadiSolver("safe", dt_max=1, rtol=1e-7, atol=1e-7),
        )

        starting_solution = sol
        sol = sim.solve(starting_solution=starting_solution, save_at_cycles=200, calc_esoh=False)

        filename = f"/{bat_id}.pkl"
        if i == 0:
            sol.save(filename)

        df = pd.DataFrame(sol.summary_variables['Measured capacity [A.h]'], columns=['Capacity'])
        df[pm_list[0]] = sei_val
        df[pm_list[1]] = plating_val
        df['Charge rate'] = crate
        df['break_point'] = break_points[i-1]

        filename = f"/{bat_id}.csv"
        if i == 1:
            df.to_csv(filename, index=False)


if __name__ == "__main__":
    xg_pred = pd.read_csv('./xgb_cap_50.csv')

    pm_combinations = [
    (row.id, row.sei, row.plating, row.crate, [int(row.b1), int(row.b2)],row.pred_scale)
    for row in xg_pred.itertuples(index=False)
]
    
    ids, sei_vals, plating_vals, crates,break_points, scale = zip(*pm_combinations)
    id = list(ids)
    pm_values_1 = np.array(sei_vals)
    pm_values_2 = np.array(plating_vals)
    crate_list  = list(crates)         
    break_points = list(break_points)   
    scale = list(scale)

    import concurrent.futures
    import logging


    with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:

        future_to_params = {executor.submit(run_simulation, *pm): pm for pm in pm_combinations[:]}

        for future in concurrent.futures.as_completed(future_to_params):
            params = future_to_params[future]
            try:
                future.result()
                logging.info(f"Completed simulation for {params}")
            except Exception as exc:
                logging.error(f"Simulation generated an exception for {params}: {exc}")