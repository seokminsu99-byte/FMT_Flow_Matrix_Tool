"""The global GIS direction action must honor an existing LASSO selection."""
import unittest
from unittest import mock

import gui
from test_workspace_v6 import make_app


class GisLassoDispatchTests(unittest.TestCase):
    def make_project(self, points=(), drawing=False):
        app = make_app([[0, 0], [0, 0]])
        app.source_kind = "gis_project"
        app.roi_polygon_points = list(points)
        app.lasso_mode = drawing
        app.clip_lasso_to_matrix = mock.Mock()
        app.convert_selected_pipe_layer_to_matrix = mock.Mock()
        return app

    def test_confirmed_lasso_routes_to_clip_not_whole_layer(self):
        app = self.make_project([(1, 1), (5, 1), (1, 5)])
        app.recalculate_gis_directions()
        app.clip_lasso_to_matrix.assert_called_once_with()
        app.convert_selected_pipe_layer_to_matrix.assert_not_called()

    def test_button_confirms_valid_draft_lasso_like_direct_clip_button(self):
        app = self.make_project([(1, 1), (5, 1), (1, 5)], drawing=True)
        app.recalculate_gis_directions()
        app.clip_lasso_to_matrix.assert_called_once_with()
        app.convert_selected_pipe_layer_to_matrix.assert_not_called()

    def test_incomplete_lasso_never_falls_back_to_whole_layer(self):
        for points, drawing in (([], True), ([(1, 1)], True), ([(1, 1), (5, 1)], False)):
            with self.subTest(points=points, drawing=drawing):
                app = self.make_project(points, drawing)
                with mock.patch.object(gui.messagebox, "showwarning") as warning:
                    app.recalculate_gis_directions()
                warning.assert_called_once()
                app.clip_lasso_to_matrix.assert_not_called()
                app.convert_selected_pipe_layer_to_matrix.assert_not_called()

    def test_no_selection_preserves_whole_layer_conversion(self):
        app = self.make_project()
        app.recalculate_gis_directions()
        app.convert_selected_pipe_layer_to_matrix.assert_called_once_with()
        app.clip_lasso_to_matrix.assert_not_called()

    def test_running_job_cannot_dispatch_either_conversion(self):
        app = self.make_project([(1, 1), (5, 1), (1, 5)])
        app._task_busy = True
        app.recalculate_gis_directions()
        app.clip_lasso_to_matrix.assert_not_called()
        app.convert_selected_pipe_layer_to_matrix.assert_not_called()


if __name__ == '__main__':
    unittest.main()
