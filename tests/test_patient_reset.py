import sys
import unittest
from unittest.mock import MagicMock, patch

from scoring import compute_gaze_score

with patch.dict(sys.modules, cv2=MagicMock(), mediapipe=MagicMock()):
    from gui_app import DementiaDetectionGUI


class PatientResetTests(unittest.TestCase):
    def test_reset_excludes_previous_gaze_and_releases_camera(self):
        app = MagicMock()
        app.gaze_running = True
        camera = app.gaze_cap
        app.gaze_results = {'trials': 10, 'accuracy': 60}
        app.trial_count, app.correct_trials = 10, 6
        app.gaze_score = compute_gaze_score(app.gaze_results)
        app.stop_gaze_experiment.side_effect = lambda: DementiaDetectionGUI.stop_gaze_experiment(app)
        DementiaDetectionGUI.reset_new_patient(app)
        self.assertFalse(compute_gaze_score(app.gaze_results)['available'])
        self.assertEqual((app.trial_count, app.correct_trials), (0, 0))
        self.assertIsNone(app.gaze_score)
        self.assertEqual(app.gaze_results['reaction_times'], [])
        camera.release.assert_called_once()
        self.assertFalse(app.gaze_running)
