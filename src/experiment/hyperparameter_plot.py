import os
import time 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.experiment.run_experiment import run_cv_for_params 
from src.utils.data_loader import load_dataset
from src.models.pin_fs_svm import PinFSSVM 



def plot_parameter_tuning_multi_type(results_dict, param_name, dataset_types, metric='auc_mean', 
                                    title=None, figsize=(12, 8), output_dir='results/plots'):
    """
    Plot charts for multiple dataset types on the same chart
    results_dict: Dictionary with key as dataset_type and value as DataFrame containing results
    """
    if not results_dict or all(df is None or df.empty for df in results_dict.values()):
        print("No results to plot")
        return None
    original_param_name = param_name
    display_param_name = param_name
    if param_name =='tau':
        x_axis_label = r"$\tau$"
        display_param_name = 'tau'
    else:
        x_axis_label = str(param_name).replace('_', ' ').title()
        display_param_name = x_axis_label
    os.makedirs(output_dir, exist_ok=True) 
    
    plt.figure(figsize=(16, 12), dpi=300)  
    
    # Colors and markers for each dataset_type
    colors = {'original': 'blue', 'noise': 'red', 'outlier': 'green', 'both': 'purple'}
    markers = {'original': 'o', 'noise': 's', 'outlier': '^', 'both': 'D'}
    legend_handles = []
    legend_texts = [] 
    
    plt.rcParams['pdf.fonttype'] = 42  
    plt.rcParams['ps.fonttype'] = 42   
    plt.rcParams['hatch.linewidth'] = 1.0  
    
    for dataset_type in dataset_types:
        if dataset_type not in results_dict or results_dict[dataset_type] is None or results_dict[dataset_type].empty:
            continue
            
        plot_df = results_dict[dataset_type].sort_values(by=original_param_name)
        
        current_color = colors.get(dataset_type, 'black')
        current_marker = markers.get(dataset_type, 'o')
        
        if str(original_param_name).upper() == 'B':
            line, = plt.plot(plot_df[original_param_name], plot_df[metric], 
                    marker=current_marker,
                    linestyle='-', 
                    color=current_color,
                    linewidth=4.5,  
                    markersize=10
                    )
        else: 
            line, = plt.plot(plot_df[original_param_name], plot_df[metric], 
                    marker=current_marker,
                    linestyle='-', 
                    color=current_color,
                    linewidth=4.5, 
                    markersize=10,  
                    )
        legend_handles.append(line)
        
        # Add standard deviation area if available
        std_col = metric.replace('_mean', '_std')
        if std_col in plot_df.columns:
            hatch_patterns = {
                'original': '...',      
                'noise': '///',         
                'outlier': '***',       
                'both': 'xxx'           
            }

            
            plt.fill_between(
            plot_df[original_param_name],
            plot_df[metric] - plot_df[std_col],
            plot_df[metric] + plot_df[std_col],
            alpha=0.15, 
            color=current_color,
            hatch=hatch_patterns.get(dataset_type, None),
            edgecolor=current_color,  
            linewidth=1.0,  
            rasterized=False  
        )
        
        legend_text_for_type = f"{dataset_type.title()}"
        
        # Mark the best point for each dataset_type
        if not plot_df[metric].empty:
            try:
                best_idx = plot_df[metric].idxmax() 
                best_x_val = plot_df.loc[best_idx, original_param_name]
                best_y_val = plot_df.loc[best_idx, metric]
                plt.scatter(best_x_val, best_y_val, color=current_color, 
                           s=250, zorder=10, edgecolor='black', marker=current_marker,
                           linewidth=3  
                           )
                legend_text_for_type += f" - Best {x_axis_label}={best_x_val:.3f} (AUC={best_y_val:.4f})"
            except ValueError: 
                print(f"Could not find best point for metric '{metric}' with {dataset_type}")
        legend_texts.append(legend_text_for_type)
    
    plt.xlabel(x_axis_label, fontsize=32, fontweight='bold') 
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=32, fontweight='bold')
    
    # Set logarithmic scale if needed
    if str(original_param_name).upper() == 'C': 
        plt.xscale('log', base=2)
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7, linewidth=2)
    elif str(original_param_name).upper() == 'B':
        plt.xscale('log')
        
        all_b_values_set = set()
        for df_val in results_dict.values(): 
            if df_val is not None and not df_val.empty:
                all_b_values_set.update(df_val[original_param_name].unique())
        
        all_b_values_sorted = sorted([b for b in list(all_b_values_set) if b > 0]) 
        
        preliminary_ticks = []
        if not all_b_values_sorted:
            pass 
        elif len(all_b_values_sorted) > 15:  
            min_val = min(all_b_values_sorted) 
            max_val = max(all_b_values_sorted)
            
            num_desired_ticks = 12 

            if max_val / min_val > 50: 
                log_ticks_ideal = np.geomspace(min_val, max_val, num=num_desired_ticks)
                selected_ticks_set = set()
                for lt_ideal in log_ticks_ideal:
                    closest = closest_value(all_b_values_sorted, lt_ideal)
                    selected_ticks_set.add(closest)
                preliminary_ticks = sorted(list(selected_ticks_set))
            else: 
                step = max(1, len(all_b_values_sorted) // num_desired_ticks)
                preliminary_ticks = all_b_values_sorted[::step]
                if all_b_values_sorted[0] not in preliminary_ticks:
                    preliminary_ticks = [all_b_values_sorted[0]] + preliminary_ticks
                if all_b_values_sorted[-1] not in preliminary_ticks:
                    preliminary_ticks.append(all_b_values_sorted[-1])
                preliminary_ticks = sorted(list(set(preliminary_ticks))) 
        else: 
            preliminary_ticks = list(all_b_values_sorted)
        
        final_tick_positions = []
        if preliminary_ticks:
            if len(preliminary_ticks) <= 7: 
                final_tick_positions = preliminary_ticks
            else: 
                MIN_TICK_RATIO_B = 1.5  

                final_tick_positions.append(preliminary_ticks[0])
                for i in range(1, len(preliminary_ticks)):
                    if preliminary_ticks[i] / final_tick_positions[-1] >= MIN_TICK_RATIO_B:
                        final_tick_positions.append(preliminary_ticks[i])
                
                if preliminary_ticks[-1] not in final_tick_positions:
                    if preliminary_ticks[-1] / final_tick_positions[-1] >= MIN_TICK_RATIO_B:
                        final_tick_positions.append(preliminary_ticks[-1])
                    elif preliminary_ticks[-1] > final_tick_positions[-1]: 
                        final_tick_positions[-1] = preliminary_ticks[-1]
        
        if final_tick_positions:
            def format_tick_label(x):
                if x >= 1000:
                    return f"{x/1000:.0f}K"  
                elif x >= 100:
                    return f"{x:.0f}"
                else:
                    return f"{x:.1f}" if x != int(x) else f"{int(x)}"
            
            tick_labels = [format_tick_label(x) for x in final_tick_positions]
            plt.xticks(final_tick_positions, tick_labels, 
                      fontsize=18,  
                      fontweight='bold',
                      )  
            
            ax = plt.gca()
            ax.tick_params(axis='x', which='major', pad=15)  
            
            # After setting ticks, iterate again to plot markers only at these tick positions for B
            for dataset_type in dataset_types:
                if dataset_type not in results_dict or results_dict[dataset_type] is None or results_dict[dataset_type].empty:
                    continue
                
                plot_df_for_markers = results_dict[dataset_type].sort_values(by=original_param_name)
                points_to_mark = plot_df_for_markers[plot_df_for_markers[original_param_name].isin(final_tick_positions)]
                
                if not points_to_mark.empty:
                    plt.scatter(points_to_mark[original_param_name], points_to_mark[metric],
                                marker=markers.get(dataset_type, 'o'),
                                color=colors.get(dataset_type, 'black'),
                                s=120,  # Tăng kích thước marker
                                zorder=5)

        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7, linewidth=2)

   
    
    plt.grid(True, linestyle='--', alpha=0.8, linewidth=2)
    
    plt.xticks(fontsize=24, fontweight = 'bold')  
    plt.yticks(fontsize=26, fontweight = 'bold')  
    
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(24)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontsize(26)
        
    if legend_handles and legend_texts: 
        plt.legend(legend_handles, legend_texts, 
                  fontsize=26, 
                  loc='best',
                  frameon=True, 
                  fancybox=True, 
                  shadow=True,
                  markerscale=2.0, 
                  framealpha=0.98,   
                  edgecolor='black',
                  prop={'weight': 'bold', 'size': 26})  
    
    plt.tight_layout(pad=5.0)  
    
    safe_title = f'comparison_{param_name}_across_types'
    
    png_path = os.path.join(output_dir, f'{safe_title}.png')
    pdf_path = os.path.join(output_dir, f'{safe_title}.pdf')
    
    try:
        plt.savefig(png_path, dpi=800, bbox_inches='tight', facecolor='white') 
        print(f"PNG chart saved to {png_path}")
        
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', 
                    backend ='pdf',
                   metadata={'Creator': 'matplotlib', 'Title': safe_title})
        print(f"PDF chart saved to {pdf_path}")
        
    except Exception as e:
        print(f"Error when saving chart: {e}")
            
    plt.close() 
    return plt.gcf()

def closest_value(values, target):
    """Find the closest value to the target in the values list"""
    return min(values, key=lambda x: abs(x - target))

def generate_comparison_plots(model_class, dataset_name, dataset_types, best_params_dict, base_output_dir):
    """
    Generate comparison charts for different dataset types
    
    Parameters:
    -----------
    model_class: class, model class (PinFSSVM)
    dataset_name: str, dataset name
    dataset_types: list, list of dataset types to compare (original, noise, outlier, both)
    best_params_dict: dict, dictionary with key as dataset_type and value as dict containing best parameters
    base_output_dir: str, output directory
    """
    # Create output directory
    plot_output_dir = os.path.join(base_output_dir, "param_comparison_plots", dataset_name, model_class.__name__)
    os.makedirs(plot_output_dir, exist_ok=True)

    # Create K-fold cross-validation 
    if dataset_name == 'colon':
        n_splits = 5
    else:
        n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Dictionary to store data for each parameter
    tau_results_dict = {}
    C_results_dict = {}
    B_results_dict = {}
    
    # Define parameter values to scan
    TAU_PLOT_VALUES = [0.1, 0.5, 1.0]
    C_PLOT_VALUES = [2**i for i in range(-3, 6)]  
    
    # Load data for each dataset_type and create charts
    print(f"Creating comparison charts for dataset {dataset_name} with types: {', '.join(dataset_types)}")
    
    # Dictionary to store X, y data for each dataset_type to avoid reloading
    datasets = {}
    for dataset_type in dataset_types:
        print(f"  Loading data for {dataset_name} (type: {dataset_type})...")
        try:
            X, y = load_dataset(dataset_name, dataset_type)
            if X.shape[0] > 0 and X.shape[1] > 0:
                datasets[dataset_type] = (X, y)
                
                max_features = X.shape[1]
                
                if dataset_type in best_params_dict:
                    best_params = best_params_dict[dataset_type]
                    
                    fixed_param_C = best_params.get('C')
                    fixed_param_B = best_params.get('B') 
                    fixed_param_tau = best_params.get('tau')

                    # --- Define B_scan_values_for_loop for the B parameter scan ---
                    current_B_scan_values = set()
                    if fixed_param_B is not None:
                        try:
                            fixed_param_B_int = int(round(float(fixed_param_B)))
                            if 1 <= fixed_param_B_int <= max_features:
                                current_B_scan_values.add(fixed_param_B_int)
                        except ValueError:
                            print(f"Warning: Could not convert fixed_param_B '{fixed_param_B}' to int for {dataset_type}")


                    if max_features > 0:
                        num_log_points = 10 
                        if max_features == 1:
                            current_B_scan_values.add(1)
                        else:
                            actual_num_log_points = min(num_log_points, max_features)
                            if actual_num_log_points > 0 : 
                                log_spaced_points = np.geomspace(1, max_features, num=actual_num_log_points, endpoint=True)
                                for p in log_spaced_points:
                                    p_int = int(round(p))
                                    if 1 <= p_int <= max_features: 
                                        current_B_scan_values.add(p_int)
                        
                        # Add some specific small feature counts if not already present
                        small_b_options = [1, 2, 5, 10, 15, 20, min(50, max_features), min(100, max_features)] 
                        for sbv in small_b_options:
                            if sbv <= max_features and sbv > 0: 
                                current_B_scan_values.add(sbv)
                        
                        current_B_scan_values.add(max_features) 

                    B_scan_values_for_loop = sorted([b for b in list(current_B_scan_values) if b > 0]) 
                    
                    # Fallback if list is empty but should not be
                    if not B_scan_values_for_loop and max_features > 0:
                        if fixed_param_B is not None:
                            try:
                                fixed_param_B_int = int(round(float(fixed_param_B)))
                                if 1 <= fixed_param_B_int <= max_features:
                                    B_scan_values_for_loop = [fixed_param_B_int]
                                else:
                                    B_scan_values_for_loop = [max_features]
                            except ValueError:
                                B_scan_values_for_loop = [max_features]
                        else:
                            B_scan_values_for_loop = [max_features]
                    elif not B_scan_values_for_loop and max_features == 0:
                         B_scan_values_for_loop = [1] 

                    # --- Scan tau with fixed C and B ---
                    print(f"  Generating tau data for {dataset_type} (fixed C={fixed_param_C}, fixed B={fixed_param_B})...")
                    results_tau = []
                    if fixed_param_C is not None and fixed_param_B is not None:
                        for tau_val in TAU_PLOT_VALUES:
                            current_params = {'C': fixed_param_C, 'B': fixed_param_B, 'tau': tau_val}
                            for p_name, p_val in best_params.items():
                                if p_name not in current_params: 
                                    current_params[p_name] = p_val
                            
                            cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                            if cv_result and cv_result['metrics']['auc']:
                                mean_auc = np.mean(cv_result['metrics']['auc'])
                                std_auc = np.std(cv_result['metrics']['auc'])
                                results_tau.append({'tau': tau_val, 'auc_mean': mean_auc, 'auc_std': std_auc})
                    
                    if results_tau:
                        tau_results_dict[dataset_type] = pd.DataFrame(results_tau)
                    
                    # --- Scan C with fixed tau and B ---
                    print(f"  Generating C data for {dataset_type} (fixed tau={fixed_param_tau}, fixed B={fixed_param_B})...")
                    results_C = []
                    if fixed_param_tau is not None and fixed_param_B is not None:
                        for c_val in C_PLOT_VALUES:
                            current_params = {'C': c_val, 'B': fixed_param_B, 'tau': fixed_param_tau}
                            for p_name, p_val in best_params.items():
                                if p_name not in current_params:
                                    current_params[p_name] = p_val
                            
                            cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                            if cv_result and cv_result['metrics']['auc']:
                                mean_auc = np.mean(cv_result['metrics']['auc'])
                                std_auc = np.std(cv_result['metrics']['auc'])
                                results_C.append({'C': c_val, 'auc_mean': mean_auc, 'auc_std': std_auc})
                    
                    if results_C:
                        C_results_dict[dataset_type] = pd.DataFrame(results_C)
                    
                    # --- Scan B with fixed C and tau ---
                    print(f"  Generating B data for {dataset_type} (fixed C={fixed_param_C}, fixed tau={fixed_param_tau})...")
                    print(f"    Scanning B values: {B_scan_values_for_loop}") 
                    results_B = []
                    if fixed_param_C is not None and fixed_param_tau is not None:
                        for b_val in B_scan_values_for_loop: 
                            current_params = {'C': fixed_param_C, 'B': b_val, 'tau': fixed_param_tau}
                            for p_name, p_val in best_params.items():
                                if p_name not in current_params:
                                    current_params[p_name] = p_val
                            
                            cv_result = run_cv_for_params(model_class, current_params, X, y, kf)
                            if cv_result and cv_result['metrics']['auc']:
                                mean_auc = np.mean(cv_result['metrics']['auc'])
                                std_auc = np.std(cv_result['metrics']['auc'])
                                avg_num_features = np.mean(cv_result['metrics']['num_features']) if 'num_features' in cv_result['metrics'] else b_val
                                results_B.append({'B': b_val, 'auc_mean': mean_auc, 'auc_std': std_auc, 'num_features_mean': avg_num_features})
                    
                    if results_B:
                        B_results_dict[dataset_type] = pd.DataFrame(results_B).dropna(subset=['auc_mean'])
            
            else:
                print(f"  Empty data for {dataset_name} ({dataset_type})")
                
        except Exception as e:
            print(f"  Error loading or processing data {dataset_name} ({dataset_type}): {e}")
    
    # Draw comparison charts for dataset types
    if tau_results_dict:
        print("Drawing tau comparison chart...")
        title = 'AUC vs $\\tau$ comparison for {} across dataset types'.format(dataset_name)
        plot_parameter_tuning_multi_type(
            tau_results_dict, 'tau', dataset_types, metric='auc_mean', 
            title=title,
            output_dir=plot_output_dir
        )
        
    if C_results_dict:
        print("Drawing C comparison chart...")
        plot_parameter_tuning_multi_type(
            C_results_dict, 'C', dataset_types, metric='auc_mean', 
            title=f'AUC vs C comparison for {dataset_name} across dataset types', 
            output_dir=plot_output_dir
        )
    
    if B_results_dict:
        print("Drawing B comparison chart...")
        plot_parameter_tuning_multi_type(
            B_results_dict, 'B', dataset_types, metric='auc_mean', 
            title=f'AUC vs B comparison for {dataset_name} across dataset types', 
            output_dir=plot_output_dir
        )
        
    print(f"Completed drawing comparison charts for {dataset_name}")

def main_compare_dataset_types():
    parser = argparse.ArgumentParser(description="Compare performance between dataset types for PinFSSVM.")
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the experiment results Excel file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base directory to save charts."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name to compare."
    )
    parser.add_argument(
        "--dataset_types",
        type=str,
        nargs='+',
        default=['original', 'noise', 'outlier', 'both'],
        help="List of dataset types to compare."
    )

    args = parser.parse_args()

    print(f"Loading results from: {args.results_file}")
    if not os.path.exists(args.results_file):
        print(f"Error: Could not find results file at {args.results_file}")
        return

    results_df_all = pd.read_excel(args.results_file)
    results_df_all.columns = results_df_all.columns.str.strip()  
    print(f"Loaded {len(results_df_all)} results from Excel.")

    target_model_name = PinFSSVM.__name__
    
    # Filter results for the specified dataset
    dataset_results = results_df_all[
        (results_df_all['Model'] == target_model_name) 

    ]
    
    if dataset_results.empty:
        print(f"No results found for dataset '{args.dataset_name}' with model '{target_model_name}'.")
        return
    
    type_mapping = {
        'Not noise': 'original',
        'Noise': 'noise',
        'Outlier': 'outlier',
        'Noise + Outlier': 'both'
    }
    # Collect best parameters for each dataset_type
    best_params_dict = {}
    for excel_type, code_type in type_mapping.items():
        if code_type in args.dataset_types:  
            type_results = dataset_results[dataset_results['Type of dataset'] == excel_type]
            if not type_results.empty:
                # Get row with highest Average AUC
                if 'Average AUC' in type_results.columns:
                    best_row = type_results.loc[type_results['Average AUC'].idxmax()]
                else:
                    # If no Average AUC column, take first row
                    best_row = type_results.iloc[0]
                
                params = {}
                for param in ['C', 'B', 'tau']:
                    if param in best_row.index:
                        params[param] = best_row[param]
                
                if params:
                    best_params_dict[code_type] = params
                    print(f"Best parameters for {code_type} ('{excel_type}' in Excel): {params}")
    
    if not best_params_dict:
        print(f"Could not find best parameters for any dataset type.")
        return
    
    print(f"Creating comparison charts for dataset {args.dataset_name} with types: {', '.join(args.dataset_types)}")
    try:
        generate_comparison_plots(
            model_class=PinFSSVM,
            dataset_name=args.dataset_name,
            dataset_types=args.dataset_types,
            best_params_dict=best_params_dict,
            base_output_dir=args.output_dir
        )
        print(f"Completed creating comparison charts for {args.dataset_name}.")
    except Exception as e:
        print(f"Error creating comparison charts: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main_compare_dataset_types()