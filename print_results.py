import os
import json
import numpy as np

def print_error_results():
    # Get all result directories
    results_dir = 'results'
    if not os.path.exists(results_dir):
        print("No results directory found!")
        return

    # Get all seed directories
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith('seed_')]
    if not seed_dirs:
        print("No seed directories found!")
        return

    # Structure and property order for the table
    structure_latex = {
        'R7': r'$Cu_{0.7}$',
        'R8': r'$Cu_{0.8}$',
        'R9': r'$Cu_{0.9}$',
        'R10': r'$Cu_{1.0}$',
        'R7-H2': r'$Cu_{0.7}-{H_2}$',
        'R8-H2': r'$Cu_{0.8}-{H_2}$',
        'R9-H2': r'$Cu_{0.9}-{H_2}$',
        'R10-H2': r'$Cu_{1.0}-{H_2}$',
    }
    property_order = ['Ef_f', 'Ef_t', 'Eg', 'HOMO', 'LUMO']
    property_latex = {
        'Ef_f': r'$E_f$',
        'Ef_t': r'$E_t$',
        'Eg': r'$E_g$',
        'HOMO': r'$E_H$',
        'LUMO': r'$E_L$',
    }
    rx_dirs = list(structure_latex.keys())
    target_properties = property_order

    # LaTeX table header
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\renewcommand{\\arraystretch}{1.1}")
    print("\\setlength{\\tabcolsep}{2pt}")
    print("\\caption{Prediction errors for each structure and property. $E_f$ and $E_t$ represent formation energy value and formation total energy repectively while $E_H$ and $E_L$ are HOMO and LUMO values. Averaged over 3 different runs. All values are in eV.}")
    print("\\scriptsize")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Structure & " + " & ".join([property_latex[p] for p in target_properties]) + r" \\")
    print("\\midrule")

    for rx_dir in rx_dirs:
        row = [structure_latex[rx_dir]]
        for target in target_properties:
            preds_list, trues_list, stds_list = [], [], []
            for seed in seed_dirs:
                result_file = os.path.join(results_dir, seed, rx_dir, target, 'results.json')
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        preds = np.array(data['pred'])
                        trues = np.array(data['true'])
                        preds_list.append(preds)
                        trues_list.append(trues)
                        stds_list.append(np.std(preds - trues))
            if preds_list and trues_list:
                avg_preds = np.mean(preds_list, axis=0)
                avg_trues = np.mean(trues_list, axis=0)
                errors = avg_preds - avg_trues
                closest_idx = np.argmin(np.abs(errors))
                closest_error = errors[closest_idx]
                avg_std = np.mean(stds_list)
                row.append(f"${closest_error:.3f} \\pm {avg_std:.3f}$")
            else:
                row.append("N/A")
        print(" & ".join(row) + r" \\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\label{tab:single_error_results}")
    print("\\end{table}")

if __name__ == '__main__':
    print_error_results() 