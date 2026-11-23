from itertools import count
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from crat_events import SessionEvents
from scoring import compute_gaze_score

with patch.dict(sys.modules, cv2=MagicMock(), mediapipe=MagicMock()):
    import gui_app
    from gui_app import DementiaDetectionGUI


class GazeTrackingTests(unittest.TestCase):
    def session(self):
        app = MagicMock()
        app.gaze_results = dict(reaction_times=[], saccade_speeds=[], v_peaks=[])
        app.gaze_events = SessionEvents()
        app._trial_speed_start = app.correct_trials = app.trial_count = 0
        app.current_stimulus = "LEFT"
        app.total_trials = 10
        app.gaze_running = True
        return app

    def test_no_face_timeouts(self):
        app = self.session()
        for _ in range(10):
            if app.gaze_running:
                DementiaDetectionGUI._complete_gaze_trial(app, None, None, face_seen=False)
        self.assertFalse(app.gaze_running)
        self.assertEqual(app.gaze_results['no_face'], 3)
        app.current_gaze_frame = None
        app.last_reaction_time = app.last_saccade_speed = 0
        with patch.object(gui_app.messagebox, 'showwarning') as warning:
            DementiaDetectionGUI.update_gaze_ui_loop(app)
            self.assertIn("face the camera", warning.call_args.args[1])
        DementiaDetectionGUI.finish_gaze_experiment(app)
        self.assertEqual(app.gaze_events.n_trials, 0)
        self.assertEqual(app.trial_count, 0)
        self.assertFalse(app.gaze_score['available'])

    def test_capture_loop_stops_after_tracking_loss(self):
        app = self.session()
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        app.gaze_cap.read.return_value = (True, frame)
        camera = app.gaze_cap
        app.face_mesh.process.return_value.multi_face_landmarks = []
        app.gaze_experiment_state = "FIXATION"
        app.fixation_time = 0
        app._complete_gaze_trial.side_effect = lambda *args, **kwargs: (
            DementiaDetectionGUI._complete_gaze_trial(app, *args, **kwargs))
        with patch.object(gui_app, 'time') as clock, patch.object(gui_app, 'cv2') as cv:
            clock.time.side_effect = count(10, 4)
            cv.flip.return_value = frame
            DementiaDetectionGUI.gaze_capture_loop(app)
        self.assertEqual(app.gaze_results['no_face'], 3)
        self.assertEqual(app.trial_count, 0)
        camera.release.assert_called_once()

    def test_minimum_valid_trials(self):
        self.assertFalse(compute_gaze_score({'trials': 4})['available'])
        self.assertTrue(compute_gaze_score({'trials': 5})['available'])

    def test_visible_responses_and_timeouts(self):
        app = self.session()
        for response in ['LEFT'] * 6 + ['RIGHT'] * 2 + [None] * 2:
            reaction = .5 if response else None
            DementiaDetectionGUI._complete_gaze_trial(app, response, reaction)
        DementiaDetectionGUI.finish_gaze_experiment(app)
        self.assertEqual(app.gaze_events.n_trials, 10)
        self.assertEqual(app.gaze_results['accuracy'], 60)
        self.assertEqual(app.gaze_score['score'], 32)
