import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.unimodal import VisionOnlyStudent, TextOnlyStudent


def test_unimodal_models():
    bs = 2
    pv = torch.randn(bs, 3, 224, 224)
    ids = torch.randint(0, 1000, (bs, 128))
    mask = torch.ones(bs, 128, dtype=torch.long)

    # 1. Test VisionOnlyStudent
    vis_model = VisionOnlyStudent(
        vision="mobilevit-xx-small",
        fusion_dim=256,
        num_modality_classes=2,
        num_location_classes=5
    )
    v_out1 = vis_model(pv)
    v_out2 = vis_model(pv, ids, mask)  # test with extra args
    assert v_out1["logits_modality"].shape == (bs, 2)
    assert v_out1["logits_location"].shape == (bs, 5)
    assert v_out2["logits_modality"].shape == (bs, 2)
    assert v_out2["logits_location"].shape == (bs, 5)
    print("[PASS] VisionOnlyStudent forward test passed.")

    # 2. Test TextOnlyStudent
    txt_model = TextOnlyStudent(
        text="bert-mini",
        fusion_dim=256,
        num_modality_classes=2,
        num_location_classes=5
    )
    t_out1 = txt_model(input_ids=ids, attention_mask=mask)
    t_out2 = txt_model(pv, ids, mask)  # test with positional args
    assert t_out1["logits_modality"].shape == (bs, 2)
    assert t_out1["logits_location"].shape == (bs, 5)
    assert t_out2["logits_modality"].shape == (bs, 2)
    assert t_out2["logits_location"].shape == (bs, 5)
    print("[PASS] TextOnlyStudent forward test passed.")


if __name__ == "__main__":
    test_unimodal_models()
    print("All unimodal tests passed!")
