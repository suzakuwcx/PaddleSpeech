import torch
from paddleaudio.audiotools.core.audio_signal import AudioSignal

from paddlespeech.t2s.modules.losses import MultiMelSpectrogramLoss
from paddlespeech.t2s.modules.losses import MultiScaleSTFTLoss


def test_dac_losses():
    for i in range(10):
        loss_origin = torch.load(f'tests/unit/tts/data/{i}-loss.pt')
        recons = AudioSignal(f'tests/unit/tts/data/{i}-recons.wav')
        signal = AudioSignal(f'tests/unit/tts/data/{i}-signal.wav')
        loss_fn_1 = MultiScaleSTFTLoss()
        loss_fn_2 = MultiMelSpectrogramLoss(
            n_mels=[5, 10, 20, 40, 80, 160, 320],
            window_lengths=[32, 64, 128, 256, 512, 1024, 2048],
            mag_weight=0.0,
            pow=1.0,
            mel_fmin=[0, 0, 0, 0, 0, 0, 0],
            mel_fmax=[None, None, None, None, None, None, None])
        #
        # Test AudioSignal
        #
        assert abs(
            loss_fn_1(recons, signal).item() - loss_origin['stft/loss']
            .item()) < 1e-5
        assert abs(
            loss_fn_2(recons, signal).item() - loss_origin['mel/loss']
            .item()) < 1e-5

        #
        # Test Tensor
        #
        assert abs(
            loss_fn_1(recons.audio_data, signal.audio_data).item() -
            loss_origin['stft/loss'].item()) < 1e-3
        assert abs(
            loss_fn_2(recons.audio_data, signal.audio_data).item() -
            loss_origin['mel/loss'].item()) < 1e-3
