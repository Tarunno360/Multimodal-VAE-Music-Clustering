import numpy as np
import pandas as pd
import pytest
import torch
from musicvae.data import align_metadata, make_splits, duplicate_groups, prepare_features
from musicvae.metrics import evaluate
from musicvae.models import MusicVAE, objective


def test_alignment_uses_id_order_not_csv_order():
    df=pd.DataFrame({"id":[7,2,9],"genre":["a","b","c"]})
    assert align_metadata(df,np.array([9,7])).genre.tolist()==["c","a"]
    with pytest.raises(ValueError):align_metadata(df,np.array([7,7]))
    with pytest.raises(ValueError):align_metadata(df,np.array([8]))


def test_duplicate_groups_cannot_cross_split():
    groups=np.repeat(np.arange(40),2)
    splits=make_splits(groups)
    sets=[set(groups[ix]) for ix in splits.values()]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert sorted(np.concatenate(list(splits.values())))==list(range(80))
    meta=pd.DataFrame({"title":["Same!","same","Other"],"lyrics":["a","b","c"]})
    gs=duplicate_groups(meta,np.arange(12).reshape(3,2,2))
    assert gs[0]==gs[1] and gs[2]!=gs[0]


def test_noise_is_excluded_and_coverage_reported():
    x=np.array([[0.],[.1],[10.],[10.1],[1e6]])
    out=evaluate(x,np.array([0,0,1,1,-1]),np.array([0,0,1,1,1]))
    ref=evaluate(x[:4],np.array([0,0,1,1]),np.array([0,0,1,1]))
    assert out["coverage"]==.8 and out["ari"]==1.
    assert out["silhouette"]==ref["silhouette"]
    assert evaluate(x,np.zeros(5),np.zeros(5))["silhouette"] is None
    assert evaluate(x,-np.ones(5),np.zeros(5))["ari"] is None
    assert evaluate(x,np.arange(5),np.arange(5))["silhouette"] is None


def test_training_transforms_do_not_learn_test_words_or_values():
    rng=np.random.default_rng(1)
    x=rng.normal(size=(20,128,256)).astype("float32")
    meta=pd.DataFrame({"lyrics":[f"common music sound melody word{i%5}" for i in range(20)]})
    splits={"train":np.arange(14),"val":np.arange(14,17),"test":np.arange(17,20)}
    _,_,original=prepare_features(x,meta,splits,text_dim=3)
    x[17:]=10000
    meta.loc[17:,"lyrics"]="onlyheldout uniqueheldouttoken"
    _,_,updated=prepare_features(x,meta,splits,text_dim=3)
    assert original["audio_mean"]==updated["audio_mean"]
    assert original["audio_std"]==updated["audio_std"]
    assert "uniqueheldouttoken" not in updated["vectorizer"].vocabulary_
    np.testing.assert_allclose(original["svd"].components_,updated["svd"].components_)


@pytest.mark.parametrize("kind",["dense","conv","multimodal","hybrid"])
def test_models_have_valid_gradients_and_signed_outputs(kind):
    torch.manual_seed(5)
    model=MusicVAE(kind=kind,latent=8,text_dim=16)
    a=torch.randn(4,1,32,64); t=torch.randn(4,16)
    out=model(a,t)
    assert out[0].shape==a.shape
    loss,*_=objective(out,a,t,.01,.05)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.mean.weight.grad.abs().sum()>0
    # A random initial bias can make every prediction positive. Test capacity
    # to represent negative targets explicitly instead of depending on that draw.
    output_layer=[m for m in model.audio_decoder.modules() if isinstance(m,(torch.nn.Linear,torch.nn.ConvTranspose2d))][-1]
    with torch.no_grad():
        output_layer.weight.zero_()
        output_layer.bias.fill_(-2.)
        negative=model(a,t)[0]
    torch.testing.assert_close(negative,torch.full_like(negative,-2.))


def test_kl_at_unit_normal_is_zero():
    x=torch.zeros(3,1,32,64); mu=torch.zeros(3,8)
    vals=objective((x,None,mu,mu),x,None,1.)
    assert all(v==0 for v in vals)


def test_conditional_model_requires_explicit_labels():
    model=MusicVAE(condition_dim=5)
    with pytest.raises(ValueError):model(torch.zeros(2,1,32,64))
