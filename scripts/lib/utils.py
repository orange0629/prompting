import os
import numpy as np
import sacrebleu
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring
from multiprocessing import Pool

def split_multi_answer(ans, sep=';', close=True, add_no_comment=False):

    """Splits string of all reference answers into a list of formatted answers"""

    answers = ans.strip().split(sep)
    split_answers = []
    for a in answers:
        a = a.strip()
        if len(a):
            if close:  # add a period after all answers
                if a[-1] != '.':
                    split_answers.append(a + '.')
                else:
                    split_answers.append(a)
            else:
                split_answers.append(a)
    # for generative metrics only, check if "I have no comment" is listed as a correct answer.
    # if not, add it.
    if add_no_comment and "I have no comment." not in split_answers:
        split_answers.append("I have no comment.")

    return split_answers

def bleurt_score(prediction, ref_true, ref_false, bleurt):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    for idx in tqdm(range(len(prediction))):
        scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
                                     references=ref_true[idx])['scores']
        scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
                                      references=ref_false[idx])['scores']
        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def tmp_func(a, b, bleurt):
    return bleurt.compute(predictions=a,references=b)['scores']

def bleurt_score_parallel(prediction, ref_true, ref_false, bleurt):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    
    pool = Pool(os.cpu_count())
    
    mp_list = [([prediction[idx]] * len(ref_true[idx]), ref_true[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_true_list = [tmp for tmp in mapping]

    mp_list = [([prediction[idx]] * len(ref_false[idx]), ref_false[idx], bleurt) for idx in range(len(prediction))]
    mapping = pool.starmap(tmp_func, mp_list)
    scores_false_list = [tmp for tmp in mapping]

    for idx in range(len(prediction)):
        #scores_true = bleurt.compute(predictions=[prediction[idx]] * len(ref_true[idx]),
        #                             references=ref_true[idx])['scores']
        #scores_false = bleurt.compute(predictions=[prediction[idx]] * len(ref_false[idx]),
        #                              references=ref_false[idx])['scores']
        scores_true = scores_true_list[idx]
        scores_false = scores_false_list[idx]

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEURT_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []

            if calc == 'max':
                res_metric[col_name].append(max(scores_true))
            elif calc == 'diff':
                res_metric[col_name].append(max(scores_true) - max(scores_false))
            elif calc == 'acc':
                res_metric[col_name].append(int(max(scores_true) > max(scores_false)))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric

def bleu_score(prediction, ref_true, ref_false):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(os.cpu_count())
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #bleu_scores = [_bleu([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_bleu, mp_list)
        bleu_scores = [tmp for tmp in mapping]
        bleu_correct = np.nanmax(bleu_scores[:len(ref_true[idx])])
        bleu_incorrect = np.nanmax(bleu_scores[len(ref_true[idx]):])

        for calc in ['max', 'diff', 'acc']:
            col_name = f'BLEU_{calc}'
            if col_name not in res_metric:
                res_metric[col_name] = []
            
            if calc == 'max':
                res_metric[col_name].append(bleu_correct)
            elif calc == 'diff':
                res_metric[col_name].append(bleu_correct - bleu_incorrect)
            elif calc == 'acc':
                res_metric[col_name].append(int(bleu_correct > bleu_incorrect))
    
    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric


def rouge_score(prediction, ref_true, ref_false):
    assert len(prediction) == len(ref_true) == len(ref_false)
    res_metric = {}
    pool = Pool(os.cpu_count())
    for idx in range(len(prediction)):
        all_refs = ref_true[idx] + ref_false[idx]
        #rouge_scores = [_rouge([ref], [prediction[idx]]) for ref in all_refs]
        mp_list = [([ref], [prediction[idx]]) for ref in all_refs]
        mapping = pool.starmap(_rouge, mp_list)
        rouge_scores = [tmp for tmp in mapping]
        
        for score_type in ['rouge1', 'rouge2', 'rougeLsum']:
            for calc in ['max', 'diff', 'acc']:
                rouge_scores_of_type = [score[score_type] for score in rouge_scores]
                rouge_correct = np.nanmax(rouge_scores_of_type[:len(ref_true[idx])])
                rouge_incorrect = np.nanmax(rouge_scores_of_type[len(ref_true[idx]):])
                col_name = f'{score_type}_{calc}'
                if col_name not in res_metric:
                    res_metric[col_name] = []
                
                if calc == 'max':
                    res_metric[col_name].append(rouge_correct)
                elif calc == 'diff':
                    res_metric[col_name].append(rouge_correct - rouge_incorrect)
                elif calc == 'acc':
                    res_metric[col_name].append(int(rouge_correct > rouge_incorrect))

    for key in res_metric:
        res_metric[key] = np.mean(res_metric[key])
    
    return res_metric


def _bleu(refs, preds):
    """
    Returns `t5` style BLEU scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L41

    :param refs:
        A `list` of `list` of reference `str`s.
    :param preds:
        A `list` of predicted `str`s.
    """
    if isinstance(refs[0], list):
        refs = [[x for x in ref] for ref in refs]
    else:
        # Need to wrap targets in another list for corpus_bleu.
        refs = [refs]
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score


def _rouge(refs, preds):
    """
    Returns `t5` style ROUGE scores. See the related implementation:
    https://github.com/google-research/text-to-text-transfer-transformer/blob/3d10afd51ba97ac29eb66ae701eca274488202f7/t5/evaluation/metrics.py#L68

    :param refs:
        A `list` of reference `strs`.
    :param preds:
        A `list` of predicted `strs`.
    """
    rouge_types = ["rouge1", "rouge2", "rougeLsum"]
    scorer = rouge_scorer.RougeScorer(rouge_types)
    # Add newlines between sentences to correctly compute `rougeLsum`.

    def _prepare_summary(summary):
        summary = summary.replace(" . ", ".\n")
        return summary

    # Accumulate confidence intervals.
    aggregator = scoring.BootstrapAggregator()
    for ref, pred in zip(refs, preds):
        ref = _prepare_summary(ref)
        pred = _prepare_summary(pred)
        aggregator.add_scores(scorer.score(ref, pred))
    result = aggregator.aggregate()
    return {type: result[type].mid.fmeasure * 100 for type in rouge_types}