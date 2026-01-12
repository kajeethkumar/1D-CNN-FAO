import numpy as np
import pywt

def dwt_denoise(ecg_signal, wavelet="db3", level=4, eps=1e-8):
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    detail = coeffs[-1]

    sigma = np.median(np.abs(detail))
    if sigma < eps:
        return ecg_signal

    sigma /= 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(ecg_signal)))

    denoised = [coeffs[0]]
    for d in coeffs[1:]:
        denoised.append(
            np.sign(d) * np.maximum(np.abs(d) - threshold, 0.0)
        )

    recon = pywt.waverec(denoised, wavelet)
    return np.nan_to_num(recon[:len(ecg_signal)])
