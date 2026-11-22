from airsense.data.preprocessing import prepare_split


def test_features_share_order_and_exclude_target_and_extras(training, rules):
    later = training[training.columns[::-1]].assign(after_close=999)
    holdout = later.drop(columns="pm25_next_day").iloc[::-1]
    later_features, labels, _ = prepare_split(later, rules, split="later")
    features, no_labels, metadata = prepare_split(holdout, rules, split="holdout")

    assert features.columns.tolist() == later_features.columns.tolist() == rules["features"]
    assert "after_close" not in features
    assert "pm25_next_day" not in later_features
    assert labels.index.equals(later_features.index)
    assert no_labels is None
    assert metadata["record_id"].tolist() == holdout["record_id"].tolist()
