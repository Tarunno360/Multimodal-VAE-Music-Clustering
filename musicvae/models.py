"""Small linear-output autoencoders for signed continuous features."""
import torch
from torch import nn


def mlp(dims):
    seq=[]
    for i,(a,b) in enumerate(zip(dims[:-1], dims[1:])):
        seq.append(nn.Linear(a,b))
        if i < len(dims)-2:
            seq.append(nn.ReLU())
    return nn.Sequential(*seq)


class MusicVAE(nn.Module):
    def __init__(self, kind="dense", latent=32, text_dim=64, condition_dim=0, deterministic=False):
        super().__init__()
        self.kind, self.latent = kind, latent
        self.text_dim, self.condition_dim = text_dim, condition_dim
        self.deterministic = deterministic
        self.multimodal = kind in ("multimodal", "hybrid")
        self.convolutional = kind in ("conv", "hybrid")
        if self.convolutional:
            self.audio_encoder = nn.Sequential(
                nn.Conv2d(1,8,3,2,1), nn.ReLU(),
                nn.Conv2d(8,16,3,2,1), nn.ReLU(),
                nn.Conv2d(16,32,3,2,1), nn.ReLU(), nn.Flatten(),
                nn.Linear(32*4*8,128), nn.ReLU())
            self.audio_decoder = nn.Sequential(nn.Linear(latent+condition_dim,32*4*8), nn.ReLU(),
                nn.Unflatten(1,(32,4,8)), nn.ConvTranspose2d(32,16,4,2,1), nn.ReLU(),
                nn.ConvTranspose2d(16,8,4,2,1), nn.ReLU(), nn.ConvTranspose2d(8,1,4,2,1))
        else:
            self.audio_encoder = nn.Sequential(nn.Flatten(), mlp([2048,256,128]), nn.ReLU())
            self.audio_decoder = nn.Sequential(mlp([latent+condition_dim,128,256,2048]), nn.Unflatten(1,(1,32,64)))
        if self.multimodal:
            self.text_encoder = nn.Sequential(mlp([text_dim,128,64]),nn.ReLU())
            self.text_decoder = mlp([latent,128,text_dim])
        self.fusion = nn.Sequential(nn.Linear(128+(64 if self.multimodal else 0)+condition_dim,128),nn.ReLU())
        self.mean = nn.Linear(128,latent)
        self.logvar = nn.Linear(128,latent)

    def encode(self, audio, text=None, condition=None):
        h=[self.audio_encoder(audio)]
        if self.multimodal:
            if text is None:
                raise ValueError("Multimodal model requires text features")
            h.append(self.text_encoder(text))
        if self.condition_dim:
            if condition is None:
                raise ValueError("Conditional model requires known conditions; it is not unsupervised")
            h.append(condition)
        h=self.fusion(torch.cat(h,dim=1))
        mu=self.mean(h)
        lv=torch.zeros_like(mu) if self.deterministic else self.logvar(h).clamp(-12,12)
        return mu,lv

    def decode(self, z, condition=None):
        audio=self.audio_decoder(torch.cat([z,condition],1) if self.condition_dim else z)
        text=self.text_decoder(z) if self.multimodal else None
        return audio,text

    def forward(self,audio,text=None,condition=None,epsilon=None):
        mu,lv=self.encode(audio,text,condition)
        if self.deterministic:
            z=mu
        else:
            eps=torch.randn_like(mu) if epsilon is None else epsilon
            z=mu+torch.exp(.5*lv)*eps
        a,t=self.decode(z,condition)
        return a,t,mu,lv


def objective(outputs,audio,text,beta,free_bits=0.,deterministic=False):
    a,t,mu,lv=outputs
    audio_mse=(a-audio).square().flatten(1).mean(1).mean()
    text_mse=(t-text).square().mean() if t is not None else audio_mse.new_zeros(())
    kl_dim=.5*(mu.square()+lv.exp()-1-lv).mean(0)
    kl=kl_dim.mean() if not deterministic else mu.new_zeros(())
    penalty=kl_dim.clamp_min(free_bits).mean() if not deterministic else kl
    # Explicit dimension-normalized objective. beta is not comparable to a sum-loss VAE.
    loss=audio_mse+text_mse+beta*penalty
    return loss, audio_mse, text_mse, kl
