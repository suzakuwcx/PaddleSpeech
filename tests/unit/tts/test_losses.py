import torch
from paddleaudio.audiotools.core.audio_signal import AudioSignal

from paddlespeech.t2s.modules.losses import MultiMelSpectrogramLoss
from paddlespeech.t2s.modules.losses import MultiScaleSTFTLoss


def test_dac_losses():
    for i in range(10):
        loss_origin = torch.load(f'tests/unit/tts/data/{i}-loss.pt')
        recons = AudioSignal(f'tests/unit/tts/data/{i}-recons.wav')
        signal = AudioSignal(f'tests/unit/tts/data/{i}-signal.wav')

        recons.audio_data.stop_gradient = False
        signal.audio_data.stop_gradient = False

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

        loss_1 = loss_fn_1(recons, signal)
        loss_1.backward()
        loss_1_grad = signal.audio_data.grad.sum()

        assert abs(
            (loss_1.item() - loss_origin['stft/loss'].item()) /
            loss_1.item()) < 1e-5, r"value incorrect for 'MultiScaleSTFTLoss'"
        assert abs(
            (loss_1_grad.item() - loss_origin['stft/grad'].sum().item()
             ) / loss_1_grad.
            item()) < 1e-5, r"gradient incorrect for 'MultiScaleSTFTLoss'"

        signal.audio_data.clear_grad()
        recons.audio_data.clear_grad()

        loss_2 = loss_fn_2(recons, signal)
        loss_2.backward()
        loss_2_grad = signal.audio_data.grad.sum()

        assert abs(
            (loss_2.item() - loss_origin['mel/loss'].item()) / loss_2.
            item()) < 1e-5, r"value incorrect for 'MultiMelSpectrogramLoss'"
        assert abs(
            (signal.audio_data.grad.sum().item() -
             loss_origin['mel/grad'].sum().item()) / loss_2_grad.
            item()) < 1e-5, r"gradient incorrect for 'MultiMelSpectrogramLoss'"

        signal.audio_data.clear_grad()
        recons.audio_data.clear_grad()

        #
        # Test Tensor
        #

        loss_1 = loss_fn_1(recons.audio_data, signal.audio_data)
        loss_1.backward()
        loss_1_grad = signal.audio_data.grad.sum()

        assert abs(loss_1.item() - loss_origin['stft/loss'].item(
        )) / loss_1.item() < 1e-5, r"value incorrect for 'MultiScaleSTFTLoss'"
        assert abs(loss_1_grad.item() - loss_origin['stft/grad'].sum()
                   .item()) / loss_1_grad.item(
                   ) < 1e-5, r"gradient incorrect for 'MultiScaleSTFTLoss'"

        signal.audio_data.clear_grad()
        recons.audio_data.clear_grad()

        loss_2 = loss_fn_2(recons.audio_data, signal.audio_data)
        loss_2.backward()
        loss_2_grad = signal.audio_data.grad.sum()

        assert abs(
            (loss_2.item() - loss_origin['mel/loss'].item()) / loss_2.
            item()) < 1e-5, r"value incorrect for 'MultiMelSpectrogramLoss'"
        assert abs(
            (loss_2_grad.item() - loss_origin['mel/grad'].sum().item()
             ) / loss_2_grad.
            item()) < 1e-5, r"gradient incorrect for 'MultiMelSpectrogramLoss'"
