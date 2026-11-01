import pytest


def test_cnn_architecture_import_and_shape():
    pytest.importorskip("torch")
    from PIL import Image

    from palmprint.models.resnet import make_resnet, preprocessing

    net = make_resnet(6, "none")
    assert net.fc.out_features == 6
    x = preprocessing(False)(Image.new("L", (128, 128)))
    assert x.shape == (3, 224, 224)
