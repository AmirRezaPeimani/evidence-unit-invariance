from evidence_unit_invariance.matching import qualified_interval_matching


def test_qualified_matching_counterexample():
    references = [(4, 28), (29, 36), (46, 47)]
    predictions = [(4, 10), (12, 36), (44, 67)]
    iou, reference_index, prediction_index = qualified_interval_matching(
        references, predictions, threshold=0.5
    )
    assert len(iou) == 1
    assert iou[0] >= 0.5
    assert len(reference_index) == len(prediction_index) == 1
