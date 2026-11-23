import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scoring import risk_level

with patch.dict(sys.modules, librosa=MagicMock(), sounddevice=MagicMock()):
    import voice_dimentia as voice


class VoiceBandTests(unittest.TestCase):
    def test_voice_result_uses_shared_bands(self):
        analyzer = MagicMock(sample_rate=10, output_dir='.')
        analyzer.record_audio.return_value = np.zeros(20)
        analyzer._feat_idx = {'jitter': 0, 'shimmer': 1, 'HNR': 2}
        features = np.zeros(21)
        analyzer.extract_features.return_value = (features, np.array([]), np.array([]), [])
        for score in (0, 19.9, 20, 35, 39.9, 40, 59.9, 60, 79.9, 80, 100):
            with self.subTest(score=score), patch.object(voice, 'log_session_for_retraining'):
                p = score / 100
                analyzer.predict.return_value = (int(p >= .6), [1-p, p], features)
                result = voice.DementiaAnalyser.run_analysis_for_gui(analyzer, plot=False)
                self.assertEqual(result['error'], '')
                self.assertEqual(result['risk_category'], risk_level(score))
