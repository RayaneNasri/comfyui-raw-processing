from typing import Any


def get_default_params() -> dict[str, Any]:
    """
    Returns the default mathematical hyperparameters for the PyTorch HDR+ pipeline.
    Stripped of all legacy rawpy configurations, file I/O flags, and execution modes.
    """
    return {
        "alignment": {
            # WARNING: these parameters are defined fine-to-coarse!
            "factors": [1, 2, 4, 4],
            "tileSizes": [16, 16, 16, 8],
            "searchRadia": [1, 4, 4, 4],
            "distances": ["L1", "L2", "L2", "L2"],
            "subpixels": [
                False,
                True,
                True,
                True,
            ],  # compute subpixel tile alignment at each level
        },
        "merging": {
            "patchSize": 16,
            "method": "DFTWiener",  # 'keepAlternate' / 'pairAverage' / 'DFTWiener'
            "noiseCurve": "exifNoiseProfile",  # 'exifNoiseProfile' / 'exifISO' / tuple (lambdaS, lambdaR)
        },
        "finishing": {
            # Note: The PyTorch pipeline expects -1 for 'auto' LTM gain estimation
            "ltmGain": -1,
            "gtmContrast": 0.075,
            "tuning": {
                "sharpenAmount": [1.0, 0.5, 0.5],
                "sharpenSigma": [1.0, 2.0, 4.0],
                "sharpenThreshold": [0.02, 0.04, 0.06],
            },
        },
    }
