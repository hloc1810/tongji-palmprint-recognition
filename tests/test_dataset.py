import pytest

from palmprint.data.dataset import build_manifest, generate_synthetic_roi, palm_id_from_name


def test_label_boundary():
    assert [palm_id_from_name(x) for x in ("00001", "00010", "00011", "06000")] == [1, 1, 2, 600]
    for value in ("00000", "06001", "thumb", "3a"):
        with pytest.raises(ValueError):
            palm_id_from_name(value)


def test_fold_integrity(tmp_path):
    generate_synthetic_roi(tmp_path / "raw", palms=5, samples=4)
    manifest = build_manifest(tmp_path / "raw", strict=False, folds=2)
    assert len(manifest) == 40
    assert not manifest.sample_uid.duplicated().any()
    train = manifest[manifest.session == "session1"]
    test = manifest[manifest.session == "session2"]
    assert train.fold.ge(0).all() and test.fold.eq(-1).all()
    assert all(set(group.fold) == {0, 1} for _, group in train.groupby("palm_id"))
    assert set(train.sample_uid).isdisjoint(test.sample_uid)
    assert (train.groupby(["palm_id", "fold"]).size() == 2).all()
    with pytest.raises(ValueError):
        build_manifest(tmp_path / "raw", strict=True)
