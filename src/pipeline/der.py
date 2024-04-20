from typing import Optional, Tuple, Dict
from pyannote.core import Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def uem_timeline_from_file(uem_file, uniq_name=''):
    """
    Generate pyannote timeline segments for uem file

     <UEM> file format
     UNIQ_SPEAKER_ID CHANNEL START_TIME END_TIME
    """
    timeline = Timeline(uri=uniq_name)
    with open(uem_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            speaker_id, channel, start_time, end_time = line.split()
            timeline.add(Segment(float(start_time), float(end_time)))

    return timeline


def score_labels(
    AUDIO_RTTM_MAP, all_reference, all_hypothesis, collar=0.25, ignore_overlap=True, verbose: bool = True
) -> Optional[Tuple[DiarizationErrorRate, Dict]]:
    """
    Calculate DER, CER, FA and MISS rate from hypotheses and references. Hypothesis results are
    coming from Pyannote-formatted speaker diarization results and References are coming from
    Pyannote-formatted RTTM data.


    Args:
        AUDIO_RTTM_MAP (dict): Dictionary containing information provided from manifestpath
        all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
        all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation
        verbose (bool): Warns if RTTM file is not found.

    Returns:
        metric (pyannote.DiarizationErrorRate): Pyannote Diarization Error Rate metric object. This object contains detailed scores of each audiofile.
        mapping (dict): Mapping dict containing the mapping speaker label for each audio input

    < Caveat >
    Unlike md-eval.pl, "no score" collar in pyannote.metrics is the maximum length of
    "no score" collar from left to right. Therefore, if 0.25s is applied for "no score"
    collar in md-eval.pl, 0.5s should be applied for pyannote.metrics.
    """
    metric = None
    if len(all_reference) == len(all_hypothesis):
        metric = DiarizationErrorRate(collar=2 * collar, skip_overlap=ignore_overlap)

        mapping_dict = {}
        for (reference, hypothesis) in zip(all_reference, all_hypothesis):
            ref_key, ref_labels = reference
            _, hyp_labels = hypothesis
            uem = AUDIO_RTTM_MAP[ref_key].get('uem_filepath', None)
            if uem is not None:
                uem = uem_timeline_from_file(uem_file=uem, uniq_name=ref_key)
            metric(ref_labels, hyp_labels, uem=uem, detailed=True)
            mapping_dict[ref_key] = metric.optimal_mapping(ref_labels, hyp_labels)

        DER = abs(metric)
        CER = metric['confusion'] / metric['total']
        FA = metric['false alarm'] / metric['total']
        MISS = metric['missed detection'] / metric['total']
        itemized_errors = (DER, CER, FA, MISS)

        print(
            "Cumulative Results for collar {} sec and ignore_overlap {}: \n FA: {:.4f}\t MISS {:.4f}\t \
                Diarization ER: {:.4f}\t, Confusion ER:{:.4f}".format(
                collar, ignore_overlap, FA, MISS, DER, CER
            )
        )

        return metric, mapping_dict, itemized_errors
    elif verbose:
        print(
            "Check if each ground truth RTTMs were present in the provided manifest file. Skipping calculation of Diariazation Error Rate"
        )
    return None
