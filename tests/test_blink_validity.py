import unittest

from dementia_analyzer import DementiaAnalyzer
from dev.verify_scores import blink_baseline
from scoring import compute_overall_score


class BlinkValidityTests(unittest.TestCase):
    def test_calibration_only(self):
        analyzer = DementiaAnalyzer()
        analyzer.detect_blink(.35, .35, timestamp=100)
        self.assertIsNone(analyzer.calculate_dementia_risk()['risk_score'])

    def test_valid_window(self):
        analyzer = DementiaAnalyzer()
        for timestamp in (100, 105, 148, 149):
            analyzer.detect_blink(.35, .35, timestamp=timestamp)
            self.assertEqual(analyzer.calculate_dementia_risk()['risk_score'],
                             28 if timestamp == 149 else None)
        self.assertAlmostEqual(blink_baseline()['risk_score'], 34.522293, places=6)

    def test_fusion_drops_missing_scores(self):
        result = compute_overall_score(gaze_score=50, voice_score=70)
        self.assertEqual(result['used'], ['gaze', 'voice'])
        self.assertEqual(result['score'], 62.3)
        self.assertEqual(compute_overall_score(voice_score=35)['weights'], {'voice': 1})
        self.assertFalse(compute_overall_score()['available'])

    def test_results_and_csv_exclude_incomplete_blink(self):
        import csv
        import sys
        from contextlib import chdir
        from tempfile import TemporaryDirectory
        from unittest.mock import MagicMock, patch

        with patch.dict(sys.modules, cv2=MagicMock(), mediapipe=MagicMock()):
            import gui_app
            from gui_adapter import GUIDementiaAnalyzer

        adapter = GUIDementiaAnalyzer.__new__(GUIDementiaAnalyzer)
        adapter.analyzer = DementiaAnalyzer()
        adapter.frame_count = 1
        adapter.analyzer.detect_blink(.35, .35)
        app = MagicMock()
        app.patient_data, app.gaze_results = {}, {}
        app.final_metrics = adapter.get_metrics()
        app.voice_results = {'risk_pct': 35, 'risk_category': 'MILD'}
        gui_app.DementiaDetectionGUI.populate_results(app)
        self.assertEqual(app.overall_score['used'], ['voice'])
        self.assertIn('not completed', app.final_metrics_text.insert.call_args.args[1])
        with TemporaryDirectory() as temp, chdir(temp), patch.object(gui_app, 'messagebox'):
            gui_app.DementiaDetectionGUI.export_patient_data(app)
            with open('patients_record.csv', newline='') as records:
                row = next(csv.DictReader(records))
        self.assertEqual(row['Eye Score (%)'], '')
        self.assertEqual(row['Eye Risk Level'], 'Not completed')
        self.assertEqual(float(row['Overall Score (%)']), 35)
