import scipy
from sed_scores_eval.utils.scores import create_score_dataframe
import sed_scores_eval


def post_process(c_scores, eval_dataloader):
    nf, nc = c_scores.size()  # (n_frame, n_classes)
    c_scores = c_scores.detach().cpu().numpy()
    scores_raw = create_score_dataframe(
            scores=c_scores,
            timestamps=[i * 10 / nf for i in range(nf + 1)],
            event_classes=eval_dataloader.dataset.classes,
    )
        
    c_scores = scipy.ndimage.filters.median_filter(c_scores, (3, 1))
    scores_postprocessed = create_score_dataframe(
        scores=c_scores,
        timestamps=[i * 10 / nf for i in range(nf + 1)],
        event_classes=eval_dataloader.dataset.classes,
    )
    return scores_raw, scores_postprocessed


def compute_psds_from_scores(
    scores,
    ground_truth_file,
    durations_file,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
    alpha_ct=0,
    alpha_st=0,
    max_efpr=100,
    num_jobs=4,
    save_dir=None,
):
    psds, psd_roc, single_class_rocs, *_ = sed_scores_eval.intersection_based.psds(
        scores=scores, 
        ground_truth=ground_truth_file,
        audio_durations=durations_file,
        dtc_threshold=dtc_threshold, 
        gtc_threshold=gtc_threshold,
        cttc_threshold=cttc_threshold, 
        alpha_ct=alpha_ct, 
        alpha_st=alpha_st,
        max_efpr=max_efpr, 
        num_jobs=num_jobs,
    )
    return psds

def compute_collar_f1(    
    scores,
    ground_truth_file,
    collar = 0.2,
    offset_collar_rate = 0.2,
    time_decimals = 30):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.collar_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        onset_collar=collar, offset_collar=collar,
        offset_collar_rate=offset_collar_rate,
        time_decimals=time_decimals,
        num_jobs=8,
    )
    return f_best

def compute_seg_f1(    
    scores,
    ground_truth_file,
    durations_file):
    
    f_best, p_best, r_best, thresholds_best, stats_best = sed_scores_eval.segment_based.best_fscore(
        scores=scores,
        ground_truth=ground_truth_file,
        audio_durations=durations_file,
        num_jobs=8,
    )
    return f_best

