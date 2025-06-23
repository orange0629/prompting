import re
import os
import glob
import json
import pandas as pd
from colorama import Fore, Style
from fastlangid.langid import LID
from collections import defaultdict
from prefill_tokens import prefill_tokens
from analyze_utils import calc_acc_v2, extract_thought_and_answer, calculate_toxic_rate

def get_thought_and_answer_langs(filepath: str) -> tuple[dict, dict]:
    """Get the language of the thought and answer from the given filepath"""
    langid = LID()
    thought_lang = defaultdict(int)
    answer_lang = defaultdict(int)
    with open(filepath) as f:
        n = 0
        for _, line in enumerate(f):
            n += 1
            data = json.loads(line)
            output = data["output"]
            thought, answer = extract_thought_and_answer(output)
            thought_lang[langid.predict(thought)] += 1
            answer_lang[langid.predict(answer)] += 1
    # print(f"Number of lines: {n}")
    # Normalize by the number of lines
    for key, value in thought_lang.items():
        thought_lang[key] = value / n
    for key, value in answer_lang.items():
        answer_lang[key] = value / n
    return thought_lang, answer_lang

def cot_vs_reasoning():
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
    target_languages = {'es', 'ru', 'ja', 'ko', 'te', 'zh-CN'}


    model_results = {'cot': {}, 'lrm': {}}
    for jsonl_filename in glob.glob("log/MATH-500/*/answer_extracted/*a3b*thinking*.jsonl"):
        if 'Okay' in jsonl_filename:
            continue

        if 'thinking_prefill' in jsonl_filename:
            prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
            lang = token2lang[prefill_phrase]
        else:
            lang = jsonl_filename.split('/')[2]

        input_lang = jsonl_filename.split('/')[2]
        if input_lang not in target_languages:
            continue
        if 'To evaluate' in jsonl_filename:
            mode = 'prefill_en'
        else:
            mode = 'prefill_native'

        if input_lang not in model_results['cot']:
            model_results['cot'][input_lang] = {}

        acc, count = calc_acc_v2(jsonl_filename)
        thought_lang, answer_lang = get_thought_and_answer_langs(jsonl_filename.replace('/answer_extracted',''))
        print(jsonl_filename)
        print("Answer languages:")
        print(f"{answer_lang.get('en', 0):.1%} / {answer_lang.get('zh-hans', 0):.1%} / {answer_lang.get(input_lang, 0):.1%} / {answer_lang.get('<unk>', 0):.1%}")
        print(acc, )
        print('------')
        model_results['cot'][input_lang][mode] = {
            'acc': acc,
            'en': answer_lang.get('en', 0),
            'zh' : answer_lang.get('zh-hans', 0),
            'native': answer_lang.get(input_lang, 0),
            lang: answer_lang.get(lang, 0),
        }
    print(model_results['cot']['ja'])
    for jsonl_filename in glob.glob("log/MATH-500/*/answer_extracted/*A3B*thinking*.jsonl"):
        if 'thinking_prefill' in jsonl_filename:
            prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
            lang = token2lang[prefill_phrase]
        else:
            lang = jsonl_filename.split('/')[2]

        input_lang = jsonl_filename.split('/')[2]
        if input_lang not in target_languages:
            continue

        if 'Okay' in jsonl_filename:
            mode = 'prefill_en'
        elif '嗯' in jsonl_filename:
            mode = 'prefill_zh'
        elif token2lang[prefill_phrase] == input_lang:
            mode = 'prefill_native'
        else:
            continue

        if input_lang not in model_results['lrm']:
            model_results['lrm'][input_lang] = {}

        acc, count = calc_acc_v2(jsonl_filename)
        thought_lang, answer_lang = get_thought_and_answer_langs(jsonl_filename.replace('/answer_extracted',''))
        print(jsonl_filename)
        print("Answer languages:")
        print(f"{answer_lang.get('en', 0):.1%} / {answer_lang.get('zh-hans', 0):.1%} / {answer_lang.get(input_lang, 0):.1%} / {answer_lang.get('<unk>', 0):.1%}")
        print(acc, )
        print('------')
        model_results['lrm'][input_lang][mode] = {
            'acc': acc,
            'en': thought_lang.get('en', 0),
            'zh' : thought_lang.get('zh-hans', 0),
            'native': thought_lang.get(input_lang, 0),
            lang: thought_lang.get(lang, 0),
        }
    print(model_results['lrm']['ja'])
    # After collecting all results, generate the LaTeX table
    lang_display_order = ['zh-CN', 'ja', 'ko', 'es', 'ru', 'te']
    lang_display_names = ['zh', 'ja', 'ko', 'es', 'ru', 'te']

    # Start LaTeX table with NeurIPS style
    latex_table = "\\begin{table}[t]\n"
    latex_table += "\\caption{Comparison of reasoning off vs. reasoning on across languages}\n"
    latex_table += "\\label{tab:reasoning_vs_cot_comparison}\n"
    latex_table += "\\centering\n"

    # Start tabular environment without vertical lines
    latex_table += "\\begin{tabular}{l" + "c"*len(lang_display_order) + "}\n"
    latex_table += "\\toprule\n"

    # Add the language header spanning all language columns
    latex_table += "& \\multicolumn{" + str(len(lang_display_order)) + "}{c}{Language} \\\\\n"
    latex_table += "\\cmidrule(r){2-" + str(len(lang_display_order)+1) + "}\n"
    latex_table += "Model Configuration & " + " & ".join(lang_display_names) + " \\\\\n"
    latex_table += "\\midrule\n"

    # Reasoning off (COT) section
    latex_table += "\\multicolumn{" + str(len(lang_display_order)+1) + "}{l}{\\textit{Qwen3 30 A3B (reasoning off)}} \\\\\n"

    # Prefill English row
    latex_table += "prefill english (To evaluate) & "
    for lang in lang_display_order:
        if lang in model_results['cot'] and 'prefill_en' in model_results['cot'][lang]:
            acc = model_results['cot'][lang]['prefill_en']['acc']
            latex_table += f"{acc:.1%} & "
        else:
            latex_table += "-- & "
    latex_table = latex_table[:-2] + "\\\\\n"

    # Prefill Native row
    latex_table += "prefill native & "
    for lang in lang_display_order:
        if lang in model_results['cot'] and 'prefill_native' in model_results['cot'][lang]:
            acc = model_results['cot'][lang]['prefill_native']['acc']
            latex_table += f"{acc:.1%} & "
        else:
            latex_table += "-- & "
    latex_table = latex_table[:-2] + "\\\\\n"

    # Add midrule between sections
    latex_table += "\\midrule\n"

    # Reasoning on (LRM) section
    latex_table += "\\multicolumn{" + str(len(lang_display_order)+1) + "}{l}{\\textit{Qwen3 30 A3B (reasoning on)}} \\\\\n"

    # Prefill English row
    latex_table += "prefill english (To evaluate) & "
    for lang in lang_display_order:
        if lang in model_results['lrm'] and 'prefill_en' in model_results['lrm'][lang]:
            acc = model_results['lrm'][lang]['prefill_en']['acc']
            latex_table += f"{acc:.1%} & "
        else:
            latex_table += "-- & "
    latex_table = latex_table[:-2] + "\\\\\n"

    # Prefill Native row
    latex_table += "prefill native & "
    for lang in lang_display_order:
        if lang in model_results['lrm'] and 'prefill_native' in model_results['lrm'][lang]:
            acc = model_results['lrm'][lang]['prefill_native']['acc']
            latex_table += f"{acc:.1%} & "
        else:
            latex_table += "-- & "
    latex_table = latex_table[:-2] + "\\\\\n"

    # Close the table with bottomrule
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}"

    print(latex_table.replace('%', '\%'))

def mmmlu_reasoning():
    model_names = [
        'R1-Distill-Llama-8B',
        'Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    ]
    languages = [
        'zh-CN','sw', 'ko', 'ja', 'en', 'es'
    ]

    lang2phrase = {token: lang for lang, token in prefill_tokens['DeepSeek-R1-Distill-Qwen-14B'].items() }
    lang2phrase['먼저'] = 'ko'
    lang2phrase['좋아'] = 'ko'
    lang2phrase['Ili kup'] = 'sw'
    # reset content
    with open('viz/mmlu_reasoning_stats.jsonl', 'w') as fout:
        pass

    for model_name in model_names:
        for input_lang in languages:
            for jsonl_filename in glob.glob(f"log/mmlu/{input_lang}/answer_extracted/*{model_name}*.jsonl"):
                if 'thinking_prefill' in jsonl_filename:
                    prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
                    lang = lang2phrase[prefill_phrase]
                else:
                    lang = jsonl_filename.split('/')[2]

                if lang not in languages:
                    continue
                thought_lang, answer_lang = get_thought_and_answer_langs(jsonl_filename.replace('answer_extracted/',''))
                subject2corrects = {}
                subject2totals = {}
                with open(jsonl_filename, 'r') as f:
                    for line in f:
                        payload = json.loads(line)
                        subject = payload['subject']
                        if subject == 'human_aging':
                            break

                        if subject not in subject2totals:
                            subject2totals[subject] = 0
                            subject2corrects[subject] = 0
                        if payload['answer'] == payload['answer_extracted']:
                            subject2corrects[subject] += 1
                        subject2totals[subject] += 1
                average_score = 0
                total_subject = 0
                outputs = {
                    'model': model_name,
                    'input_lang': input_lang,
                    'reasoning_lang': lang,
                    'reasoning_stats': {
                        'en': thought_lang.get('en', 0),
                        'zh' : thought_lang.get('zh-hans', 0),
                        'native': thought_lang.get(lang, 0),
                    },
                    'answer_stats': {
                        'en': answer_lang.get('en', 0),
                        'zh' : answer_lang.get('zh-hans', 0),
                        'native': answer_lang.get(input_lang, 0),
                    },
                    'subject_accuracy': {}
                }

                for subject, corr in subject2corrects.items():
                    acc = corr/subject2totals[subject]
                    outputs['subject_accuracy'][subject] = acc
                    average_score += acc
                    total_subject += 1
                outputs['total_subjects'] = total_subject
                outputs['average'] = average_score/total_subject
                with open('viz/mmlu_reasoning_stats.jsonl', 'a') as fout:
                    fout.write(json.dumps(outputs)+'\n')


def generate_neurips_math_table(all_results, target_languages, model_name_mapping, lang_order=None):
    """
    Generate a NeurIPS-style LaTeX table comparing prefill_en vs prefill_native
    across different languages.
    """
    # Language display names for better readability
    lang_display = {
        'es': 'Spanish',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'te': 'Telugu',
        'zh-CN': 'Chinese',
        'sw': 'Swahili'
    }
    if lang_order is None:
        lang_order = sorted(target_languages)
    else:
        # Ensure all languages in lang_order are in target_languages
        for lang in lang_order:
            if lang not in target_languages:
                print(f"Warning: Language '{lang}' in lang_order is not in target_languages")
        
        # Add any missing languages from target_languages
        for lang in target_languages:
            if lang not in lang_order:
                print(f"Warning: Language '{lang}' from target_languages not in lang_order")
                lang_order.append(lang)
    
    
    # Start LaTeX table
    print("\\begin{table}")
    print("\\centering")
    print("\\caption{Comparison of English vs. Native prefill strategies on MATH-500 across languages}")
    print("\\label{tab:math-500-prefill-comparison}")
    
    # Create column specification
    col_spec = "l" + "c" * len(lang_order)
    print('\\small')
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    
    # Column headers (Mode + Languages)
    header = "Strategy & " + " & ".join([lang_display.get(lang, lang) for lang in lang_order]) + " \\\\"
    print(header)
    print("\\midrule")
    
    # Combined results across models
    combined_results = {model: {} for model in all_results}
    
    # Process each model
    for model_name, results in all_results.items():
        # Add model name as section header
        display_name = model_name_mapping.get(model_name, model_name)
        print(f"\\multicolumn{{{len(lang_order) + 1}}}{{c}}{{\\textbf{{{display_name}}}}} \\\\")
        
        # Organize results by mode and language
        mode_lang_acc = {}
        for r in results:
            mode = r['mode']
            lang = r['input_lang']
            acc = r['acc']
            
            if mode not in mode_lang_acc:
                mode_lang_acc[mode] = {}
            
            mode_lang_acc[mode][lang] = acc
        
        # Store organized results
        combined_results[model_name] = mode_lang_acc
        
        # Print rows for each strategy
        mode_display = {
            'prefill_en': 'Prefill English',
            'prefill_native': 'Prefill Input Language',
            'baseline': 'Baseline'
        }
        
        for mode in ['prefill_en', 'prefill_native', 'baseline']:
            if mode in mode_lang_acc:
                row = f"{mode_display.get(mode, mode)} & "
                for lang in lang_order:
                    if lang in mode_lang_acc[mode]:
                        row += f"{mode_lang_acc[mode][lang]:.1%} & "
                    else:
                        row += "-- & "
                row = row.rstrip(" & ") + " \\\\"
                print(row.replace('%', '\%'))
        
        # Calculate and display difference (prefill_en - prefill_native)
        if 'prefill_en' in mode_lang_acc and 'prefill_native' in mode_lang_acc:
            row = "Difference (EN - Input) & "
            for lang in lang_order:
                if lang in mode_lang_acc['prefill_en'] and lang in mode_lang_acc['prefill_native']:
                    diff = mode_lang_acc['prefill_en'][lang] - mode_lang_acc['prefill_native'][lang]
                    # Highlight positive differences (EN better than Native)
                    if diff > 0:
                        row += f"\\textbf{{{diff:+.1%}}} & "
                    else:
                        row += f"{diff:+.1%} & "
                else:
                    row += "-- & "
            row = row.rstrip(" & ") + " \\\\"
            print(row.replace('%', '\%'))
        
        print("\\midrule")
    
    # Add average section across all models
    print(f"\\multicolumn{{{len(lang_order) + 1}}}{{c}}{{\\textbf{{Average across all models}}}} \\\\")    
    # Calculate average performance for each mode and language
    avg_results = {}
    for mode in ['prefill_en', 'prefill_native', 'baseline']:
        avg_results[mode] = {}
        for lang in lang_order:
            values = []
            for model, mode_data in combined_results.items():
                if mode in mode_data and lang in mode_data[mode]:
                    values.append(mode_data[mode][lang])
            if values:
                avg_results[mode][lang] = sum(values) / len(values)

    # Print average rows
    for mode in ['prefill_en', 'prefill_native', 'baseline']:
        if mode in avg_results:
            row = f"{mode_display.get(mode, mode)} & "
            for lang in lang_order:
                if lang in avg_results[mode]:
                    row += f"{avg_results[mode][lang]:.1%} & "
                else:
                    row += "-- & "
            row = row.rstrip(" & ") + " \\\\"
            print(row.replace('%', '\%'))
    
    # Calculate and display average difference
    if 'prefill_en' in avg_results and 'prefill_native' in avg_results:
        row = "Difference (EN - Input) & "
        for lang in lang_order:
            if lang in avg_results['prefill_en'] and lang in avg_results['prefill_native']:
                diff = avg_results['prefill_en'][lang] - avg_results['prefill_native'][lang]
                # Highlight positive differences
                if diff > 0:
                    row += f"\\textbf{{{diff:+.1%}}} & "
                else:
                    row += f"{diff:+.1%} & "
            else:
                row += "-- & "
        row = row.rstrip(" & ") + " \\\\"
        print(row.replace('%', '\%'))
    
    # End LaTeX table
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

def math_table():
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
    target_languages = {'es', 'ru', 'ja', 'ko', 'te', 'zh-CN', 'sw'}
    
    model_names = [
        'R1-Distill-Llama-8B',
        'Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    ]
    model_name_mapping = {
        'R1-Distill-Llama-8B': 'DeepSeek-R1-Distill-Llama-8B',
        'QwQ-32B': 'QwQ-32B', # Example display name
        'Qwen3-30B-A3B': 'Qwen3-30B-A3B',
        'Qwen-14B': 'DeepSeek-R1-Distill-Qwen-14B',
        # 'super_model_v3_final': 'SuperNet v3',
    }
    all_results = {}
    for model_name in model_names:
        model_results = []
        for jsonl_filename in glob.glob(f"log/MATH-500/*/answer_extracted/*{model_name}*thinking*.jsonl"):
            if 'thinking_prefill' in jsonl_filename:
                prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
                lang = token2lang[prefill_phrase]
            else:
                lang = jsonl_filename.split('/')[2]

            input_lang = jsonl_filename.split('/')[2]
            if input_lang not in target_languages:
                continue

            if 'Okay' in jsonl_filename and input_lang != 'en':
                mode = 'prefill_en'
            elif '嗯' in jsonl_filename and input_lang != 'zh-CN':
                mode = 'prefill_zh'
            elif token2lang[prefill_phrase] == input_lang:
                mode = 'prefill_native'
            else:
                continue

            acc, count = calc_acc_v2(jsonl_filename)
            if count <= 100:
                continue

            thought_lang, answer_lang = get_thought_and_answer_langs(jsonl_filename.replace('/answer_extracted',''))
            print(jsonl_filename)
            print("Answer languages:")
            print(f"{answer_lang.get('en', 0):.1%} / {answer_lang.get('zh-hans', 0):.1%} / {answer_lang.get(input_lang, 0):.1%} / {answer_lang.get('<unk>', 0):.1%}")
            print(acc, )
            print('------')
            model_results.append({
                'input_lang': input_lang,
                'control': lang,
                'acc': acc,
                'mode': mode,
                'en': thought_lang.get('en', 0),
                'zh' : thought_lang.get('zh-hans', 0),
                'native': thought_lang.get(input_lang, 0),
                lang: thought_lang.get(lang, 0),
            })
        for jsonl_filename in glob.glob(f"log/MATH-500/*/answer_extracted/*{model_name}.jsonl"):
            lang = jsonl_filename.split('/')[2]

            input_lang = jsonl_filename.split('/')[2]
            if input_lang not in target_languages:
                continue
            mode = 'baseline'
            acc, count = calc_acc_v2(jsonl_filename)
            if count <= 100:
                continue

            thought_lang, answer_lang = get_thought_and_answer_langs(jsonl_filename.replace('/answer_extracted',''))
            print(jsonl_filename)
            print("Answer languages:")
            print(f"{answer_lang.get('en', 0):.1%} / {answer_lang.get('zh-hans', 0):.1%} / {answer_lang.get(input_lang, 0):.1%} / {answer_lang.get('<unk>', 0):.1%}")
            print(acc, )
            print('------')
            model_results.append({
                'input_lang': lang,
                'control': None,
                'acc': acc,
                'mode': mode,
                'en': thought_lang.get('en', 0),
                'zh' : thought_lang.get('zh-hans', 0),
                'native': thought_lang.get(input_lang, 0),
                lang: thought_lang.get(lang, 0),
            })
        all_results[model_name] = model_results
    
    generate_neurips_math_table(all_results, target_languages, model_name_mapping, lang_order=['zh-CN', 'es', 'ru', 'sw', 'ja', 'te', 'ko'])

def generate_neurips_mmlu_reasoning_comparison_table():
    """
    Generate a NeurIPS-style LaTeX table comparing MMLU performance when:
    1. Reasoning language is the same as input language
    2. Reasoning language is English
    """
    target_languages = {'en', 'zh-CN', 'es', 'sw', 'ja', 'ko'}
    language_order = ['en', 'zh-CN', 'es', 'sw', 'ja', 'ko']
    
    # Language display names for better readability
    lang_display = {
        'en': 'English',
        'es': 'Spanish',
        'zh-CN': 'Chinese',
        'sw': 'Swahili',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    # Group results by model, input language, and reasoning language
    model_results = {}
    
    with open('viz/mmlu_reasoning_stats.jsonl', 'r') as f:
        for line in f:
            row = json.loads(line)
            model = row['model']
            input_lang = row['input_lang']
            reasoning_lang = row['reasoning_lang']
            
            if model not in model_results:
                model_results[model] = {}
            
            if input_lang not in model_results[model]:
                model_results[model][input_lang] = {}
            
            model_results[model][input_lang][reasoning_lang] = row
    
    # Start LaTeX table
    print("\\begin{table}")
    print("\\centering")
    print("\\caption{Comparison of MMLU performance when reasoning in native language vs. English}")
    print("\\label{tab:mmlu-reasoning-language-comparison}")
    
    # Create column specification
    col_spec = "l" + "c" * len(language_order)
    print('\\small')
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")
    
    # Column headers (Strategy + Languages)
    header = "Strategy & " + " & ".join([lang_display.get(lang, lang) for lang in language_order]) + " \\\\"
    print(header)
    print("\\midrule")
    
    # Collect results for calculating averages later
    combined_results = {model: {} for model in model_results}
    
    # Process each model
    for model_name, lang_data in model_results.items():
        # Add model name as section header
        print(f"\\multicolumn{{{len(language_order) + 1}}}{{c}}{{\\textbf{{{model_name}}}}} \\\\")
        
        # Organize results by reasoning strategy and input language
        strategy_lang_acc = {
            'native_reasoning': {},
            'english_reasoning': {}
        }
        
        for input_lang in language_order:
            if input_lang in lang_data:
                # Find entries with native reasoning vs. English reasoning
                for reasoning_lang, row in lang_data[input_lang].items():
                    if reasoning_lang == input_lang:
                        strategy_lang_acc['native_reasoning'][input_lang] = row['average']
                    elif reasoning_lang == 'en':
                        strategy_lang_acc['english_reasoning'][input_lang] = row['average']
        
        # Store organized results
        combined_results[model_name] = strategy_lang_acc
        
        # Print rows for each strategy
        strategy_display = {
            'native_reasoning': 'Prefill Input Language',
            'english_reasoning': 'Prefill English'
        }
        
        for strategy in ['native_reasoning', 'english_reasoning']:
            row = f"{strategy_display[strategy]} & "
            for lang in language_order:
                if lang in strategy_lang_acc[strategy]:
                    row += f"{strategy_lang_acc[strategy][lang]:.1%} & "
                else:
                    row += "-- & "
            row = row.rstrip(" & ") + " \\\\"
            print(row.replace('%', '\%'))
        
        # Calculate and display difference (English - Native)
        row = "Difference (EN - Input) & "
        for lang in language_order:
            if (lang in strategy_lang_acc['english_reasoning'] and 
                lang in strategy_lang_acc['native_reasoning']):
                diff = strategy_lang_acc['english_reasoning'][lang] - strategy_lang_acc['native_reasoning'][lang]
                # Highlight positive differences (EN better than Native)
                if diff > 0:
                    row += f"\\textbf{{{diff:+.1%}}} & "
                else:
                    row += f"{diff:+.1%} & "
            else:
                row += "-- & "
        row = row.rstrip(" & ") + " \\\\"
        print(row.replace('%', '\%'))
        
        print("\\midrule")
    
    # Add average section across all models
    print(f"\\multicolumn{{{len(language_order) + 1}}}{{c}}{{\\textbf{{Average across all models}}}} \\\\")
    
    # Calculate average performance for each strategy and language
    avg_results = {
        'native_reasoning': {},
        'english_reasoning': {}
    }
    
    for strategy in ['native_reasoning', 'english_reasoning']:
        for lang in language_order:
            values = []
            for model, strategy_data in combined_results.items():
                if lang in strategy_data[strategy]:
                    values.append(strategy_data[strategy][lang])
            if values:
                avg_results[strategy][lang] = sum(values) / len(values)
    
    # Print average rows
    for strategy in ['native_reasoning', 'english_reasoning']:
        row = f"{strategy_display[strategy]} & "
        for lang in language_order:
            if lang in avg_results[strategy]:
                row += f"{avg_results[strategy][lang]:.1%} & "
            else:
                row += "-- & "
        row = row.rstrip(" & ") + " \\\\"
        print(row.replace('%', '\%'))
    
    # Calculate and display average difference
    row = "Difference (EN - Native) & "
    for lang in language_order:
        if (lang in avg_results['english_reasoning'] and 
            lang in avg_results['native_reasoning']):
            diff = avg_results['english_reasoning'][lang] - avg_results['native_reasoning'][lang]
            # Highlight positive differences
            if diff > 0:
                row += f"\\textbf{{{diff:+.1%}}} & "
            else:
                row += f"{diff:+.1%} & "
        else:
            row += "-- & "
    row = row.rstrip(" & ") + " \\\\"
    print(row.replace('%', '\%'))
    
    # End LaTeX table
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def get_reasoning_seed_phrase():
    from transformers import AutoTokenizer
    top_n = 5
    dataset_name = "MATH-500"
    model_names = [
        'R1-Distill-Llama-8B',
        'Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    ]
    model_name2tokenizer_name = {
        'R1-Distill-Llama-8B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'Qwen-14B': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'QwQ-32B': 'Qwen/QwQ-32B',
        'Qwen3-30B-A3B': 'Qwen/Qwen3-30B-A3B',        
    }
    IGNORE_TOKENS = 0
    top_n = 4
    double_decode = False
    valid_languages = {'en', 'zh', 'ja', 'es', 'ru', 'zh-CN', 'ko', 'sw', 'te'}
    for model_name in model_names:
        for jsonl_filename in glob.glob(f"log/{dataset_name}/*/*{model_name}.jsonl"):
            input_lang = jsonl_filename.split('/')[2]
            if input_lang not in valid_languages:
                continue
            print('Model: ', model_name)
            print('Language', input_lang)
            first_n_token_string_count = {i: defaultdict(int) for i in range(top_n)}
            first_n_token_combination_count = {i: defaultdict(int) for i in range(top_n)}
            tokenizer = AutoTokenizer.from_pretrained(model_name2tokenizer_name[model_name])
            with open(jsonl_filename, 'r') as f:
                for line in f:
                    row = json.loads(line)
                    token2logprob_seq = row["logprobs"]
                    if len(token2logprob_seq) != 0:
                        string = ""
                        tokens = []
                        for i in range(IGNORE_TOKENS, top_n):
                            token2logprob = token2logprob_seq[i]
                            if type(token2logprob) == dict:
                                token, _ = list(token2logprob.items())[0]
                            elif type(token2logprob) == list:
                                token = token2logprob[0]["token"]
                            else:
                                raise TypeError(f"Unexpected type: {type(token2logprob)}")
                            if double_decode:
                                token_id = tokenizer.convert_tokens_to_ids(token)
                                tokens.append(token_id)
                            string += token
                            first_n_token_string_count[i][string] += 1
                            first_n_token_combination_count[i]['-'.join([ str(r) for r in tokens])] += 1
            for i in range(IGNORE_TOKENS, top_n):
                print(f"Top {i + 1 - IGNORE_TOKENS} tokens:")
                for string, count in sorted(first_n_token_string_count[i].items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"{string}: {count}")
                if double_decode:
                    for token_string, count in sorted(first_n_token_combination_count[i].items(), key=lambda x: x[1], reverse=True)[:10]:
                        token_str_list = token_string.split('-')
                        tokens = [ int(s) for s in token_str_list]
                        string = tokenizer.decode(tokens)
                        print(f"{string}: {count}")
                print()


def generate_latex_table(model_results):
    # Define language order and display names
    lang_order = ['zh', 'es', 'ru', 'sw', 'ja', 'te', 'ko']
    lang_display = {
        'zh': 'Chinese',
        'zh-hans': 'Chinese',
        'zh-CN': 'Chinese',
        'es': 'Spanish',
        'ru': 'Russian',
        'sw': 'Swahili',
        'ja': 'Japanese',
        'te': 'Telugu',
        'ko': 'Korean'
    }
    
    # Define models and their display names
    model_order = [
        'R1-Distill-Llama-8B',
        'Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B'
    ]
    model_display = {
        'R1-Distill-Llama-8B': 'DeepSeek-R1-Distill-Llama-8B',
        'Qwen-14B': 'DeepSeek-R1-Distill-Qwen-14B',
        'QwQ-32B': 'QwQ-32B',
        'Qwen3-30B-A3B': 'Qwen3-30B-A3B'
    }

    # Organize data by model, language, and prefill type
    data = defaultdict(lambda: defaultdict(dict))
    
    for result in model_results:
        # Extract model name directly from result
        model_name = None
        for m in model_order:
            if m in result.get('model_name', ''):
                model_name = m
                break
                
        if not model_name:
            # Try to extract model from any available field
            for field, value in result.items():
                if isinstance(value, str):
                    for m in model_order:
                        if m in value:
                            model_name = m
                            break
                    if model_name:
                        break
                        
        if not model_name:
            continue
            
        input_lang = result['input_lang']
        if input_lang == 'zh-CN' or input_lang == 'zh-hans':
            input_lang = 'zh'
            
        mode = result['mode']
        asr = result['asr'] * 100  # Convert to percentage
        
        if mode == 'prefill_en':
            data[model_name][input_lang]['en'] = asr
        elif mode == 'prefill_native':
            data[model_name][input_lang]['native'] = asr
    
    # Calculate averages across models
    averages = defaultdict(lambda: {'en': 0, 'native': 0, 'count': 0})
    
    for model, langs in data.items():
        for lang, values in langs.items():
            if 'en' in values and 'native' in values:
                averages[lang]['en'] += values['en']
                averages[lang]['native'] += values['native']
                averages[lang]['count'] += 1
    
    for lang in averages:
        if averages[lang]['count'] > 0:
            averages[lang]['en'] /= averages[lang]['count']
            averages[lang]['native'] /= averages[lang]['count']
    
    # Start generating the LaTeX table
    latex = []
    latex.append(r"\begin{table}")
    latex.append(r"\caption{Comparison of English vs. Native prefill strategies on toxic content generation across languages ordered by speakers population.}")
    latex.append(r"\label{tab:toxic-prefill}")
    latex.append(r"\centering")
    latex.append(r"\begin{tabular}{lccccccc}")
    latex.append(r"\toprule")
    
    # Header row
    header = ["Strategy"] + [lang_display.get(lang, lang) for lang in lang_order]
    latex.append(" & ".join(header) + r" \\")
    latex.append(r"\midrule")
    
    # Process each model
    for model in model_order:
        model_display_name = model_display.get(model, model)
        latex.append(r"\multicolumn{8}{c}{" + model_display_name + r"} \\")
        
        # English Prefill row
        en_values = []
        for lang in lang_order:
            val = data[model].get(lang, {}).get('en', float('nan'))
            en_values.append(f"{val:.1f}\%" if not pd.isna(val) else "N/A")
        latex.append("Prefill English & " + " & ".join(en_values) + r" \\")
        
        # Native Prefill row
        native_values = []
        for lang in lang_order:
            val = data[model].get(lang, {}).get('native', float('nan'))
            native_values.append(f"{val:.1f}\%" if not pd.isna(val) else "N/A")
        latex.append("Prefill Input Language & " + " & ".join(native_values) + r" \\")
        
        # Difference row
        diff_values = []
        for lang in lang_order:
            en_val = data[model].get(lang, {}).get('en', float('nan'))
            native_val = data[model].get(lang, {}).get('native', float('nan'))
            
            if not pd.isna(en_val) and not pd.isna(native_val):
                diff = en_val - native_val
                sign = "+" if diff > 0 else ""
                diff_values.append(f"{sign}{diff:.1f}\%")
            else:
                diff_values.append("N/A")
        
        latex.append("Difference (EN - Input) & " + " & ".join(diff_values) + r" \\")
        
        # Add extra space between models
        if model != model_order[-1]:
            latex.append(r"\midrule")
    
    # Average across all models
    latex.append(r"\midrule")
    latex.append(r"\multicolumn{8}{c}{Average across all models} \\")
    
    # Average English Prefill
    avg_en_values = []
    for lang in lang_order:
        val = averages.get(lang, {}).get('en', float('nan'))
        avg_en_values.append(f"{val:.1f}\%" if not pd.isna(val) else "N/A")
    latex.append("English Prefill & " + " & ".join(avg_en_values) + r" \\")
    
    # Average Native Prefill
    avg_native_values = []
    for lang in lang_order:
        val = averages.get(lang, {}).get('native', float('nan'))
        avg_native_values.append(f"{val:.1f}\%" if not pd.isna(val) else "N/A")
    latex.append("Prefill Input Language & " + " & ".join(avg_native_values) + r" \\")
    
    # Average Difference
    avg_diff_values = []
    for lang in lang_order:
        en_val = averages.get(lang, {}).get('en', float('nan'))
        native_val = averages.get(lang, {}).get('native', float('nan'))
        
        if not pd.isna(en_val) and not pd.isna(native_val):
            diff = en_val - native_val
            sign = "+" if diff > 0 else ""
            avg_diff_values.append(f"{sign}{diff:.1f}\%")
        else:
            avg_diff_values.append("N/A")
    
    latex.append("Difference (EN - Input) & " + " & ".join(avg_diff_values) + r" \\")
    
    # Finish the table
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)

def get_toxic_results():
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
    target_languages = {'es', 'ru', 'ja', 'ko', 'te', 'zh', 'sw'}
    answer_src = 'answer_extracted_v2'
    model_names = [
        'R1-Distill-Llama-8B',
        'Qwen-14B',
        'QwQ-32B',
        'Qwen3-30B-A3B',
    ]
    model_name_mapping = {
        'R1-Distill-Llama-8B': 'DeepSeek-R1-Distill-Llama-8B',
        'QwQ-32B': 'QwQ-32B', # Example display name
        'Qwen3-30B-A3B': 'Qwen3-30B-A3B',
        'Qwen-14B': 'DeepSeek-R1-Distill-Qwen-14B',
    }
    model_results = []
    for model_name in model_names:
        for language in target_languages:
            # print(f"log/toxic_bench/{language}/{answer_src}/*{model_name}*.jsonl")
            for jsonl_filename in glob.glob(f"log/toxic_bench/{language}/{answer_src}/*{model_name}*thinking*.jsonl"):
                if 'thinking_prefill' in jsonl_filename:
                    prefill_phrase = jsonl_filename.split('thinking_prefill-')[-1].replace('.jsonl','')
                    lang = token2lang[prefill_phrase]
                else:
                    lang = jsonl_filename.split('/')[2]

                input_lang = jsonl_filename.split('/')[2]
                if input_lang not in target_languages:
                    continue

                if 'Okay' in jsonl_filename and input_lang != 'en':
                    mode = 'prefill_en'
                # elif '嗯' in jsonl_filename and input_lang != 'zh':
                #     mode = 'prefill_zh'
                elif token2lang[prefill_phrase] == input_lang or \
                    (input_lang == 'zh' and token2lang[prefill_phrase] == 'zh-CN'):
                    mode = 'prefill_native'
                else:
                    continue

                toxic = calculate_toxic_rate(jsonl_filename)
                model_results.append({
                    'input_lang': input_lang,
                    'control': lang,
                    'asr': toxic[0],
                    'mode': mode,
                    'model_name': model_name,
                    # 'en': thought_lang.get('en', 0),
                    # 'zh' : thought_lang.get('zh-hans', 0),
                    # 'native': thought_lang.get(input_lang, 0),
                    # lang: thought_lang.get(lang, 0),
                })
    print(model_results[0])
    print(len(model_results))
    latex_table = generate_latex_table(model_results)
    print(latex_table)




if __name__ == "__main__":
    # mmmlu_reasoning()
    # cot_vs_reasoning()
    math_table()
    # generate_neurips_mmlu_reasoning_comparison_table()
    # get_toxic_results()