import os
import re
import json
import glob
import logging
import numpy as np
from collections import Counter
from prefill_tokens import prefill_tokens
from analyze_utils import calc_acc_v2
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as path_effects

token2lang = {}
for model_name, lang2token_mapping in prefill_tokens.items():
    for lang, prefill in lang2token_mapping.items():
        token2lang[prefill] = lang
token2lang['Ili Kup'] = 'sw'
token2lang['To evaluate'] = 'en'
token2lang['题目'] = 'zh-hans'
token2lang['与えられた問題'] = 'ja'
token2lang['Para encontrar'] = 'es'
token2lang['ఇక్కడ'] = 'te'
token2lang['Для'] = 'ru'
token2lang['주어진'] = 'ko'

REASONING_TYPES = ["Subgoal setting", "Backtracking", "Verification", "Backward chaining"]

def parse_steps(step_string):
    # Extract step types using regex
    pattern = r'<step_\d+><([^>]+)></step_\d+>'
    step_types = re.findall(pattern, step_string)
    # Count occurrences of each step type
    return dict(Counter(step_types))

def normalize_step_types(input_string):
    # Define mappings for different type representations
    type_mapping = {
        # Direct types
        "Others": "Others",
        "Subgoal setting": "Subgoal setting",
        "Verification": "Verification", 
        "Backtracking": "Backtracking",
        "Backward chaining": "Backward chaining",
        
        # Numbered types (based on definition order)
        "type_name_1": "Subgoal setting",
        "type_name_2": "Backtracking",
        "type_name_3": "Verification",
        "type_name_4": "Backward chaining",
        "type_name_5": "Others",
        
        # Named types with prefix
        "type_name_Others": "Others",
        "type_name_Subgoal setting": "Subgoal setting",
        "type_name_Verification": "Verification",
        "type_name_Backtracking": "Backtracking",
        "type_name_Backward chaining": "Backward chaining"
    }
    
    # Process input string to extract type-value pairs
    lines = input_string.strip().split('\n')
    normalized_data = {}
    
    for line in lines:
        # Split each line into type and value
        parts = line.rsplit(' ', 1)
        if len(parts) == 2:
            type_name, value = parts
            value = float(value)
            # Normalize the type name
            if type_name in type_mapping:
                normalized_type = type_mapping[type_name]
                # Convert value to float
                try:
                    value = float(value)
                    # Add to the normalized data (summing values for same types)
                    if normalized_type in normalized_data:
                        normalized_data[normalized_type] += value
                    else:
                        normalized_data[normalized_type] = value
                except ValueError:
                    pass
                    # logging.warning(f"Warning: Could not convert value '{value}' to float for type '{type_name}'")
            elif '<Others' in type_name:
                value = float(value)
                normalized_type = 'Others'
                if normalized_type in normalized_data:
                    normalized_data[normalized_type] += value
                else:
                    normalized_data[normalized_type] = value
            else:
                for rt in REASONING_TYPES:
                    if rt.lower() in type_name.lower():
                        if rt in normalized_data:
                            normalized_data[rt] += value
                        else:
                            normalized_data[rt] = value
                        break
                # logging.warning(f"Warning: Unknown type '{type_name}'")
    
    return normalized_data

def process_steps_with_normalization(step_string):
    """
    Function that chains parse_steps and normalize_step_types together
    """
    # Step 1: Parse the steps and get counts
    step_counts = parse_steps(step_string)
    
    # Step 2: Convert the dictionary to a string format for normalize_step_types
    input_for_normalize = "\n".join([f"{step_type} {count}" for step_type, count in step_counts.items()])
    
    # Step 3: Normalize the step types
    normalized_counts = normalize_step_types(input_for_normalize)
    
    return normalized_counts


def generate_correlation_table(input_pivot, prefill_pivot=None, combined=False):
    """
    Generate a NeurIPS-style LaTeX table for correlation results between
    languages and reasoning behaviors.
    
    Parameters:
    -----------
    input_pivot : pandas.DataFrame
        Pivot table with correlation and p-value data for input languages
        Structure: MultiIndex columns with (correlation/p_value, reasoning_type)
    prefill_pivot : pandas.DataFrame, optional
        Pivot table with correlation and p-value data for prefill languages
    combined : bool, default=False
        If True, combines both input and prefill tables into one table
        If False, generates separate tables
        
    Returns:
    --------
    str
        LaTeX table code
    """
    # Language display names
    language_names = {
        'en': 'English',
        'zh-hans': 'Chinese',
        'ja': 'Japanese', 
        'es': 'Spanish',
        'te': 'Telugu',
        'ru': 'Russian',
        'ko': 'Korean',
        'sw': 'Swahili'
    }
    
    # Reasoning type display names
    reasoning_types = input_pivot.columns.levels[1]
    reasoning_display = {
        'Subgoal setting': 'Subgoal',
        'Backtracking': 'Backtrack',
        'Verification': 'Verify',
        'Backward chaining': 'Backward'
    }
    
    # Start building LaTeX code
    latex_lines = []
    
    if combined and prefill_pivot is not None:
        # Combined table for both input and prefill languages
        latex_lines.append("\\begin{table}[t]")
        latex_lines.append("\\caption{Correlation between languages and reasoning behaviors}")
        latex_lines.append("\\label{tab:lang_corr}")
        latex_lines.append("\\centering")
        latex_lines.append("\\small")
        
        # Column definition
        col_spec = "llcccc"  # Language type, Language name, and 4 reasoning types
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\toprule")
        
        # Header row
        header_row = "Type & Language"
        for rt in reasoning_types:
            display_name = reasoning_display.get(rt, rt)
            header_row += f" & {display_name}"
        header_row += " \\\\"
        latex_lines.append(header_row)
        latex_lines.append("\\midrule")
        
        # Process input languages
        latex_lines.append("\\multirow{" + str(len(input_pivot.index)) + "}{*}{Input} & \\multicolumn{5}{l}{} \\\\")
        
        first_input = True
        for lang in input_pivot.index:
            if not first_input:
                latex_lines.append("& ")
            first_input = False
            
            row = language_names.get(lang, lang)
            for rt in reasoning_types:
                corr_val = input_pivot.loc[lang, ('correlation', rt)]
                p_val = input_pivot.loc[lang, ('p_value', rt)]
                
                # Format with significance markers
                sig_marker = ""
                if p_val < 0.001:
                    sig_marker = "^{***}"
                elif p_val < 0.01:
                    sig_marker = "^{**}"
                elif p_val < 0.05:
                    sig_marker = "^{*}"
                
                formatted_val = f"{corr_val:.2f}{sig_marker}"
                row += f" & {formatted_val}"
            row += " \\\\"
            latex_lines.append(row)
        
        # Add separator before prefill languages
        latex_lines.append("\\midrule")
        
        # Process prefill languages
        latex_lines.append("\\multirow{" + str(len(prefill_pivot.index)) + "}{*}{Prefill} & \\multicolumn{5}{l}{} \\\\")
        
        first_prefill = True
        for lang in prefill_pivot.index:
            if not first_prefill:
                latex_lines.append("& ")
            first_prefill = False
            
            row = language_names.get(lang, lang)
            for rt in reasoning_types:
                corr_val = prefill_pivot.loc[lang, ('correlation', rt)]
                p_val = prefill_pivot.loc[lang, ('p_value', rt)]
                
                # Format with significance markers
                sig_marker = ""
                if p_val < 0.001:
                    sig_marker = "^{***}"
                elif p_val < 0.01:
                    sig_marker = "^{**}"
                elif p_val < 0.05:
                    sig_marker = "^{*}"
                
                formatted_val = f"{corr_val:.2f}{sig_marker}"
                row += f" & {formatted_val}"
            row += " \\\\"
            latex_lines.append(row)
    
    else:
        # Separate tables for input and prefill languages
        
        # Input language table
        latex_lines.append("\\begin{table}[t]")
        latex_lines.append("\\caption{Correlation between input languages and reasoning behaviors}")
        latex_lines.append("\\label{tab:input_lang_corr}")
        latex_lines.append("\\centering")
        
        # Column definition
        col_spec = "l" + "c" * len(reasoning_types)
        latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        latex_lines.append("\\toprule")
        
        # Header row
        header_row = "Language"
        for rt in reasoning_types:
            display_name = reasoning_display.get(rt, rt)
            header_row += f" & {display_name}"
        header_row += " \\\\"
        latex_lines.append(header_row)
        latex_lines.append("\\midrule")
        
        # Process each language
        for lang in input_pivot.index:
            row = language_names.get(lang, lang)
            for rt in reasoning_types:
                corr_val = input_pivot.loc[lang, ('correlation', rt)]
                p_val = input_pivot.loc[lang, ('p_value', rt)]
                
                # Format with significance markers
                sig_marker = ""
                if p_val < 0.001:
                    sig_marker = "^{***}"
                elif p_val < 0.01:
                    sig_marker = "^{**}"
                elif p_val < 0.05:
                    sig_marker = "^{*}"
                
                formatted_val = f"{corr_val:.2f}{sig_marker}"
                row += f" & {formatted_val}"
            row += " \\\\"
            latex_lines.append(row)
        
        # End the input language table
        latex_lines.append("\\bottomrule")
        latex_lines.append("\\multicolumn{" + str(len(reasoning_types) + 1) + "}{l}{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        # Add prefill language table if provided
        if prefill_pivot is not None:
            # Add some space
            latex_lines.append("")
            latex_lines.append("% Prefill Language Table")
            latex_lines.append("")
            
            # Prefill language table
            latex_lines.append("\\begin{table}[t]")
            latex_lines.append("\\caption{Correlation between prefill languages and reasoning behaviors}")
            latex_lines.append("\\label{tab:prefill_lang_corr}")
            latex_lines.append("\\centering")
            
            # Column definition (same as input language table)
            latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
            latex_lines.append("\\toprule")
            
            # Header row (same as input language table)
            latex_lines.append(header_row)
            latex_lines.append("\\midrule")
            
            # Process each language
            for lang in prefill_pivot.index:
                row = language_names.get(lang, lang)
                for rt in reasoning_types:
                    corr_val = prefill_pivot.loc[lang, ('correlation', rt)]
                    p_val = prefill_pivot.loc[lang, ('p_value', rt)]
                    
                    # Format with significance markers
                    sig_marker = ""
                    if p_val < 0.001:
                        sig_marker = "^{***}"
                    elif p_val < 0.01:
                        sig_marker = "^{**}"
                    elif p_val < 0.05:
                        sig_marker = "^{*}"
                    
                    formatted_val = f"{corr_val:.2f}{sig_marker}"
                    row += f" & {formatted_val}"
                row += " \\\\"
                latex_lines.append(row)
            
            # End the prefill language table
            latex_lines.append("\\bottomrule")
            latex_lines.append("\\multicolumn{" + str(len(reasoning_types) + 1) + "}{l}{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}")
            latex_lines.append("\\end{tabular}")
            latex_lines.append("\\end{table}")
    
    # Return the LaTeX code
    return "\n".join(latex_lines)


if __name__ == "__main__":
    full_data = []
    for jsonl_filename in glob.glob('log/MATH-500/*/reasoning_kind_results/*/segmentation_results/*.jsonl'):
        input_lang = jsonl_filename.split('/')[2]
        if 'thinking' not in jsonl_filename:
            continue
        action_stats = {}
        with open(jsonl_filename,'r') as f:
            for line in f:
                row = json.loads(line)
                answer = row['output'].split('[Final answer]')[-1]
                
                parsed_step = process_steps_with_normalization(answer)
                for step, count in parsed_step.items():
                    if step not in action_stats:
                        action_stats[step] = []
                    action_stats[step].append(count)
        reasoning = ["Subgoal setting", "Backtracking", "Verification", "Backward chaining"]
        reasoning_data = {}
        for reason in reasoning:
            if reason in action_stats:
                reasoning_data[reason] = np.mean(action_stats[reason])
            else:
                reasoning_data[reason] = 0.0
        model_name = jsonl_filename.split('segmentation_results/reasoning-by-')[-1].split('__thinking_')[0]
        
        prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].split('__')[0]
        reasoning_lang = token2lang[prefill_phrase]

        acc_jsonl_result = list(glob.glob(f'log/MATH-500/{input_lang}/answer_extracted/*{model_name}*thinking*{prefill_phrase}.jsonl'))[0]
        acc, count = calc_acc_v2(acc_jsonl_result)
        if count <= 100:
            continue
        print(acc)

        full_data.append({
            'input_lang': input_lang,
            'prefill_lang': reasoning_lang,
            'acc': acc,
            **reasoning_data
        })
    print(full_data[0])
    # TODO:
    # analysis the pearson corr between input/prefill languages for these 4 ["Subgoal setting", "Backtracking", "Verification", "Backward chaining"] behavior

    df = pd.DataFrame(full_data)

    # Define reasoning types
    reasoning_types = ["Subgoal setting", "Backtracking", "Verification", "Backward chaining"]

    # Get unique languages
    input_languages = df['input_lang'].unique()
    prefill_languages = df['prefill_lang'].unique()

    # Results containers
    input_lang_results = []
    prefill_lang_results = []

    # For each reasoning type
    for reasoning_type in reasoning_types:
        # For each input language
        for lang in input_languages:
            # Create a dummy variable (1 if this language is used as input, 0 otherwise)
            is_input_lang = (df['input_lang'] == lang).astype(int)
            
            # Calculate correlation with reasoning behavior
            correlation, p_value = stats.pearsonr(is_input_lang, df[reasoning_type])
            
            input_lang_results.append({
                'language': lang,
                'reasoning_type': reasoning_type,
                'correlation': correlation,
                'p_value': p_value
            })
            
        # For each prefill language
        for lang in prefill_languages:
            # Create a dummy variable (1 if this language is used as prefill, 0 otherwise)
            is_prefill_lang = (df['prefill_lang'] == lang).astype(int)
            
            # Calculate correlation with reasoning behavior
            correlation, p_value = stats.pearsonr(is_prefill_lang, df[reasoning_type])
            
            prefill_lang_results.append({
                'language': lang,
                'reasoning_type': reasoning_type,
                'correlation': correlation,
                'p_value': p_value
            })

    # Convert results to DataFrames
    input_lang_df = pd.DataFrame(input_lang_results)
    prefill_lang_df = pd.DataFrame(prefill_lang_results)

    # Create pivot tables for more readable results
    input_pivot = pd.pivot_table(
        input_lang_df, 
        values=['correlation', 'p_value'], 
        index=['language'],
        columns=['reasoning_type']
    )

    prefill_pivot = pd.pivot_table(
        prefill_lang_df, 
        values=['correlation', 'p_value'], 
        index=['language'],
        columns=['reasoning_type']
    )

    print("\nInput Language Correlations (Pivot Table):")
    print(input_pivot)
    print("\nPrefill Language Correlations (Pivot Table):")
    print(prefill_pivot)

       # Generate separate tables
    print("SEPARATE TABLES:")
    print(generate_correlation_table(input_pivot, prefill_pivot, combined=False))

    # Set up visualization style
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    
    # 1. Create a correlation heatmap between languages and reasoning behaviors
    def create_correlation_heatmap(pivot_df, title, filename, xlabel=None, ylabel=None, fontsize_multiplier=1.2):
        plt.figure(figsize=(12, 10))
        
        # Extract just correlation values
        corr_values = pivot_df.loc[:, ('correlation', slice(None))]
        corr_values.columns = corr_values.columns.droplevel(0)
        
        # Create heatmap with larger font sizes for annotations
        sns.heatmap(corr_values, annot=True, cmap='RdBu_r', center=0, 
                    vmin=-1, vmax=1, linewidths=.5, fmt='.2f',
                    annot_kws={"size": 18 * fontsize_multiplier})
        
        # Add axis labels if provided
        # if xlabel:
        #     plt.xlabel(xlabel, fontsize=17 * fontsize_multiplier)
        if ylabel:
            plt.ylabel(ylabel, fontsize=17 * fontsize_multiplier)
        plt.xlabel('')
            
        # plt.title(title, fontsize=16 * fontsize_multiplier)
        
        # Increase tick label font size
        # plt.tick_params(axis='both', which='major', labelsize=16 * fontsize_multiplier)
        plt.tick_params(axis='y', which='major', labelsize=16 * fontsize_multiplier)
        plt.tick_params(axis='x', which='major', labelsize=16 * fontsize_multiplier)  # Larger font for x-axis values    
        plt.xticks(rotation=-20, ha='center')
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))  # Save PDF version
        plt.close()
    
    # 2. Create plots showing most relevant behavior for each language
    def create_most_relevant_behavior_plot(pivot_df, title, filename, figsize=(12, 11), fontsize=25):
        # Extract correlation values
        corr_values = pivot_df.loc[:, ('correlation', slice(None))]
        corr_values.columns = corr_values.columns.droplevel(0)
        
        # Find most relevant behavior (highest absolute correlation) for each language
        most_relevant = pd.DataFrame(index=corr_values.index)
        most_relevant['behavior'] = corr_values.abs().idxmax(axis=1)
        most_relevant['correlation'] = [
            corr_values.loc[lang, behavior] 
            for lang, behavior in zip(most_relevant.index, most_relevant['behavior'])
        ]
        
        # Get p-values for the most relevant behaviors
        p_values = pivot_df.loc[:, ('p_value', slice(None))]
        p_values.columns = p_values.columns.droplevel(0)
        most_relevant['p_value'] = [
            p_values.loc[lang, behavior] 
            for lang, behavior in zip(most_relevant.index, most_relevant['behavior'])
        ]
        
        # Sort by absolute correlation value
        most_relevant = most_relevant.sort_values(by='correlation', key=abs, ascending=False)
        
        # Create the plot with specified figsize
        plt.figure(figsize=figsize)
        
        # Bar colors based on statistical significance
        colors = ['darkred' if p <= 0.01 else 'orangered' if p <= 0.05 else 'lightgrey' 
                for p in most_relevant['p_value']]
        
        bars = plt.barh(most_relevant.index, most_relevant['correlation'], color=colors)
        
        # Add behavior labels on the bars
        for i, (idx, row) in enumerate(most_relevant.iterrows()):
            # Determine text color based on correlation
            if row['p_value'] <= 0.01:  # For highly significant results (darkred bars)
                text_color = '#FFFFFF'  # Pure white for dark red bars
            else:
                text_color = 'white' if abs(row['correlation']) > 0.3 else 'black'
                
            # Make text more visible with a subtle outline effect
            plt.text(
                0.2, i, f" {row['behavior']} ", 
                ha='center', va='center', 
                color=text_color,
                fontweight='bold',
                fontsize=fontsize * 0.9,  # Adjusted relative to main fontsize
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')
                ] if row['p_value'] <= 0.01 else []
            )
        
        # Remove significance markers from bars since they'll be explained in legend
        # Instead, we'll use a clearer approach in the legend
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlim(-1, 1)
        plt.xlabel('Correlation Coefficient', fontsize=fontsize)
        plt.ylabel('Prefill Target Language', fontsize=fontsize)
        # plt.title(title, fontsize=fontsize * 1.2)
        
        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=fontsize * 0.8)
        
        # Add an enhanced legend that explains significance markers
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', label='Highly significant (p ≤ 0.01)'),
            Patch(facecolor='orangered', label='Significant (p ≤ 0.05)'),
            Patch(facecolor='lightgrey', label='Not significant (p > 0.05)')
        ]
        legend = plt.legend(handles=legend_elements, loc='upper left', fontsize=fontsize * 0.7)
        
        # Add caption below the plot explaining the figure
        caption = (
            "Figure shows the behavior with the strongest correlation for each language. "
            "Bar colors indicate statistical significance levels."
        )
        # plt.figtext(0.5, 0.01, caption, ha='center', fontsize=fontsize * 0.6)
        print(caption)
        
        plt.tight_layout()
        # Leave some space at the bottom for the caption
        plt.subplots_adjust(bottom=0.1)
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))  # Save PDF version
    
    # 3. Create a visualization showing behavior profiles for each language
    def create_language_behavior_profiles(df, lang_column, title, filename):
        reasoning_types = ["Subgoal setting", "Backtracking", "Verification", "Backward chaining"]
        
        # Get top languages by number of occurrences
        top_languages = df[lang_column].value_counts().head(6).index.tolist()
        
        plt.figure(figsize=(15, 10))
        
        # For each top language, create a bar plot for each reasoning type
        for i, language in enumerate(top_languages):
            language_data = df[df[lang_column] == language]
            
            # Create a subplot
            plt.subplot(2, 3, i+1)
            
            # Calculate mean values for each reasoning type
            means = [language_data[rt].mean() for rt in reasoning_types]
            
            # Create the bar plot
            bars = plt.bar(reasoning_types, means, color='steelblue')
            
            # Add accuracy label
            acc = language_data['acc'].mean()
            plt.text(
                0.5, 0.9, 
                f'Avg Accuracy: {acc:.2f}', 
                ha='center', 
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
            
            plt.title(f'Language: {language}')
            plt.ylabel('Average Occurrence')
            plt.xticks(rotation=45)
            plt.ylim(0, df[reasoning_types].values.max() * 1.1)  # Consistent y scale
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))  # Save PDF version
        plt.close()
    
    # 4. Create a visualization showing relationship between reasoning types and accuracy
    def create_reasoning_accuracy_plot(df, title, filename):
        plt.figure(figsize=(16, 12))
        
        for i, reasoning_type in enumerate(reasoning_types):
            plt.subplot(2, 2, i+1)
            
            # Create scatter plot for each input language
            for lang in input_languages:
                lang_data = df[df['input_lang'] == lang]
                plt.scatter(
                    lang_data[reasoning_type], 
                    lang_data['acc'], 
                    label=f'Input: {lang}',
                    alpha=0.7,
                    s=80,
                    marker='o'
                )
            
            # Add trend line
            sns.regplot(
                x=reasoning_type, 
                y='acc', 
                data=df,
                scatter=False,
                line_kws={"color": "black", "alpha": 0.5, "lw": 2}
            )
            
            plt.title(f'Accuracy vs. {reasoning_type}')
            plt.xlabel(f'Average {reasoning_type} Occurrence')
            plt.ylabel('Accuracy')
            
            # Add correlation stats
            corr, p = stats.pearsonr(df[reasoning_type], df['acc'])
            plt.annotate(
                f'r = {corr:.2f}, p = {p:.3f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                backgroundcolor='white',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8)
            )
        
        plt.suptitle(title, fontsize=20)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig(filename)
        plt.savefig(filename.replace('.png', '.pdf'))  # Save PDF version
        plt.close()

    # Create output directory if it doesn't exist
    output_dir = "viz/behaviours"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create all visualizations
    create_correlation_heatmap(
        input_pivot, 
        'Correlation between Input Languages and Reasoning Types',
        f'{output_dir}/input_language_correlation_heatmap.png',
        xlabel='Reasoning Type',
        ylabel='Input Language'
    )

    create_correlation_heatmap(
        prefill_pivot, 
        'Correlation Matrix: Prefill Target Language vs. Reasoning Type',
        f'{output_dir}/prefill_language_correlation_heatmap.png',
        xlabel='Reasoning Type',
        ylabel='Prefill Target Language',
        fontsize_multiplier=1.3  # Increase font size by 30%
    )

    create_most_relevant_behavior_plot(
        input_pivot,
        'Most Relevant Reasoning Behavior by Input Language',
        f'{output_dir}/input_language_most_relevant_behavior.png'
    )

    create_most_relevant_behavior_plot(
        prefill_pivot,
        'Most Relevant Reasoning Behavior by Target Language',
        f'{output_dir}/prefill_language_most_relevant_behavior.png',
        figsize=(12, 10),  # Increase height for better side-by-side display with heatmap
        # fontsize_multiplier=1.3  # Increase font size by 30%
    )

    create_language_behavior_profiles(
        df, 
        'input_lang', 
        'Reasoning Behavior Profiles by Input Language',
        f'{output_dir}/input_language_behavior_profiles.png'
    )

    create_language_behavior_profiles(
        df, 
        'prefill_lang', 
        'Reasoning Behavior Profiles by Prefill Language',
        f'{output_dir}/prefill_language_behavior_profiles.png'
    )

    create_reasoning_accuracy_plot(
        df,
        'Relationship Between Reasoning Behaviors and Performance',
        f'{output_dir}/reasoning_vs_accuracy.png'
    )

    print("All visualizations have been created successfully in both PNG and PDF formats!")