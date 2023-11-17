
# EmotiVoice - Multi-Voice and Prompt-Controlled TTS Engine.

# https://github.com/netease-youdao/EmotiVoice


# Quickstart
"""
EmotiVoice Docker image

The easiest way to try EmotiVoice is by running the docker image.
You need a machine with a NVidia GPU. If you have not done so, set up NVidia container toolkit by following the instructions for Linux or Windows WSL2.
Then EmotiVoice can be run with,

docker run -dp 127.0.0.1:8501:8501 syq163/emoti-voice:latest

Now open your browser and navigate to http://localhost:8501 to start using EmotiVoice's powerful TTS capabilities.
Full installation

conda create -n EmotiVoice python=3.8 -y
conda activate EmotiVoice
pip install torch torchaudio
pip install numpy numba scipy transformers==4.26.1 soundfile yacs g2p_en jieba pypinyin
"""

"""
This code is modified from https://github.com/espnet/espnet.
"""

import torch
import torch.nn as nn


from models.prompt_tts_modified.modules.encoder import Encoder
from models.prompt_tts_modified.modules.variance import DurationPredictor, VariancePredictor
from models.prompt_tts_modified.modules.alignment import AlignmentModule, GaussianUpsampling, viterbi_decode, average_by_duration
from models.prompt_tts_modified.modules.initialize import initialize

class PromptTTS(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        
        self.encoder = Encoder(
            attention_dim=config.model.encoder_n_hidden,
            attention_heads=config.model.encoder_n_heads,
            linear_units=config.model.encoder_n_hidden * 4,            
            num_blocks=config.model.encoder_n_layers,
            dropout_rate=config.model.encoder_p_dropout,
            positional_dropout_rate=config.model.encoder_p_dropout,
            attention_dropout_rate=config.model.encoder_p_dropout,
            normalize_before=True,
            concat_after=False,
            positionwise_conv_kernel_size=config.model.encoder_kernel_size_conv_mod,
            stochastic_depth_rate=0.0,
        )

        self.decoder = Encoder(
            attention_dim=config.model.decoder_n_hidden,
            attention_heads=config.model.decoder_n_heads,
            linear_units=config.model.decoder_n_hidden * 4,            
            num_blocks=config.model.decoder_n_layers,
            dropout_rate=config.model.decoder_p_dropout,
            positional_dropout_rate=config.model.decoder_p_dropout,
            attention_dropout_rate=config.model.decoder_p_dropout,
            normalize_before=True,
            concat_after=False,
            positionwise_conv_kernel_size=config.model.decoder_kernel_size_conv_mod,
            stochastic_depth_rate=0.0,
        )

        self.duration_predictor = DurationPredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=config.model.duration_n_layers,
            n_chans=config.model.variance_n_hidden,
            kernel_size=config.model.duration_kernel_size,
            dropout_rate=config.model.duration_p_dropout,
        )

        self.pitch_predictor = VariancePredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=config.model.variance_n_layers, #pitch_predictor_layers=5 in paddlespeech fs2
            n_chans=config.model.variance_n_hidden,
            kernel_size=config.model.variance_kernel_size, #pitch_predictor_kernel_size=5 in paddlespeech fs2
            dropout_rate=config.model.variance_p_dropout, #pitch_predictor_dropout=0.5 in paddlespeech fs2
        )
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size, #pitch_embed_kernel_size=1 in paddlespeech fs2
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout), #pitch_embed_dropout=0.0
        )
        self.energy_predictor = VariancePredictor(
            idim=config.model.encoder_n_hidden,
            n_layers=2,
            n_chans=config.model.variance_n_hidden,
            kernel_size=3,
            dropout_rate=config.model.variance_p_dropout,
        )
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=config.model.encoder_n_hidden,
                kernel_size=config.model.variance_embed_kernel_size,
                padding=(config.model.variance_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(config.model.variance_embde_p_dropout),
        )

        self.length_regulator = GaussianUpsampling()
        self.alignment_module = AlignmentModule(config.model.encoder_n_hidden, config.n_mels)

        self.to_mel = nn.Linear(
            in_features=config.model.decoder_n_hidden, 
            out_features=config.n_mels,
        )


        self.spk_tokenizer = nn.Embedding(config.n_speaker, config.model.encoder_n_hidden)
        self.src_word_emb = nn.Embedding(config.n_vocab, config.model.encoder_n_hidden)
        self.embed_projection1 = nn.Linear(config.model.encoder_n_hidden * 2 + config.model.bert_embedding * 2, config.model.encoder_n_hidden)

        initialize(self, "xavier_uniform")

    def forward(self, inputs_ling, input_lengths, inputs_speaker, inputs_style_embedding , inputs_content_embedding, mel_targets=None, output_lengths=None, pitch_targets=None, energy_targets=None, alpha=1.0):
        
        B = inputs_ling.size(0)
        T = inputs_ling.size(1)
        src_mask = self.get_mask_from_lengths(input_lengths)
        token_embed = self.src_word_emb(inputs_ling)
        x, _ = self.encoder(token_embed, ~src_mask.unsqueeze(-2))
        speaker_embedding = self.spk_tokenizer(inputs_speaker)
        x = torch.concat([x, speaker_embedding.unsqueeze(1).expand(B, T, -1), inputs_style_embedding.unsqueeze(1).expand(B, T, -1), inputs_content_embedding.unsqueeze(1).expand(B, T, -1)], dim=-1)
        x = self.embed_projection1(x)

        if mel_targets is not None:
            log_p_attn = self.alignment_module(text=x, feats=mel_targets.transpose(1,2), text_lengths=input_lengths, feats_lengths=output_lengths, x_masks=src_mask)
            ds, bin_loss = viterbi_decode(log_p_attn, input_lengths, output_lengths)

            ps = average_by_duration(ds, pitch_targets.squeeze(-1), input_lengths, output_lengths)
            es = average_by_duration(ds, energy_targets.squeeze(-1), input_lengths, output_lengths)

        p_outs = self.pitch_predictor(x, src_mask.unsqueeze(-1))
        e_outs = self.energy_predictor(x, src_mask.unsqueeze(-1))

        if mel_targets is not None:
            d_outs = self.duration_predictor(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(ps.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.unsqueeze(-1).transpose(1, 2)).transpose(1, 2)

        else:
            log_p_attn, ds, bin_loss, ps, es =None, None, None, None, None
            d_outs = self.duration_predictor.inference(x, src_mask.unsqueeze(-1))
            p_embs = self.pitch_embed(p_outs.unsqueeze(1)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.unsqueeze(1)).transpose(1, 2)

        x = x + p_embs + e_embs

        if mel_targets is not None:
            h_masks_upsampling = self.make_non_pad_mask(output_lengths).to(x.device) 
            x = self.length_regulator(x, ds, h_masks_upsampling, ~src_mask, alpha=alpha)
            h_masks = self.make_non_pad_mask(output_lengths).unsqueeze(-2).to(x.device)

        else:
            x = self.length_regulator(x, d_outs, None, ~src_mask)
            mel_lenghs = torch.sum(d_outs, dim=-1).int()
            # h_masks=make_non_pad_mask(mel_lenghs).unsqueeze(-2).to(x.device)
            h_masks=None
        x, _ = self.decoder(x, h_masks)
        x = self.to_mel(x)
        
        return {
            "mel_targets":mel_targets,
            "dec_outputs": x, 
            "postnet_outputs": None, 
            "pitch_predictions": p_outs.squeeze(),
            "pitch_targets": ps,
            "energy_predictions": e_outs.squeeze(),
            "energy_targets": es,
            "log_duration_predictions": d_outs,
            "duration_targets": ds,
            "input_lengths": input_lengths,
            "output_lengths": output_lengths,
            "log_p_attn": log_p_attn,
            "bin_loss": bin_loss,
        }
    def get_mask_from_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = lengths.shape[0]
        max_len = torch.max(lengths).item()
        ids = (
            torch.arange(0, max_len, device=lengths.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
        return mask
    def average_utterance_prosody(
        self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        lengths = ((~src_mask) * 1.0).sum(1)
        u_prosody_pred = u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)
        return u_prosody_pred
    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print(f"{name} is not loaded")

    def make_pad_mask(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).int()

        ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
            batch_size, -1)
        mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

        return mask


    def make_non_pad_mask(self, length, max_len=None):
        return ~self.make_pad_mask(length, max_len)


"""
This code is modified from https://github.com/jik876/hifi-gan.
"""


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

LRELU_SLOPE = 0.1

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.initial_channel, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class Discriminator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.msd = MultiScaleDiscriminator()
        self.mpd = MultiPeriodDiscriminator()

    def forward(self, y, y_hat):
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_hat)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_hat)

        return y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g, y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
