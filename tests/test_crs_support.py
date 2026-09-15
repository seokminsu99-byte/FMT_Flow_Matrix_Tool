import queue
import unittest
from unittest import mock

from pyproj import CRS

import crs_support
import gis_matrix
import gis_project
import gui


class CRSSupportTests(unittest.TestCase):
    def test_epsg_definition_reports_authority_and_linear_unit(self) -> None:
        definition = crs_support.parse_crs("EPSG:5186")

        self.assertEqual(definition.label, "EPSG:5186")
        self.assertTrue(definition.is_projected)
        self.assertEqual(definition.unit_type, "meter")
        self.assertAlmostEqual(float(definition.unit_to_meter), 1.0)

    def test_semantic_comparison_accepts_wkt1_and_wkt2_for_same_crs(self) -> None:
        wkt1 = CRS.from_epsg(5186).to_wkt(version="WKT1_ESRI")
        wkt2 = CRS.from_epsg(5186).to_wkt(version="WKT2_2019")

        self.assertTrue(crs_support.crs_equivalent(wkt1, wkt2))

    def test_vector_transform_round_trip_preserves_coordinates(self) -> None:
        original = [[[(126.9780, 37.5665), (127.0250, 37.5000)]]]
        projected = crs_support.transform_nested_points(original, "EPSG:4326", "EPSG:5186")
        restored = crs_support.transform_nested_points(projected, "EPSG:5186", "EPSG:4326")

        self.assertGreater(projected[0][0][0][0], 100_000.0)
        for actual, expected in zip(restored[0][0], original[0][0]):
            self.assertAlmostEqual(actual[0], expected[0], places=7)
            self.assertAlmostEqual(actual[1], expected[1], places=7)

    def test_polyline_reprojection_preserves_heights_and_slope_reference(self) -> None:
        record = gis_matrix.GISPolyline(
            parts=[[(126.98, 37.56), (126.99, 37.55)]],
            start_height=12.5,
            end_height=11.0,
            slope_reference_parts=[[(126.98, 37.56), (126.99, 37.55)]],
        )

        transformed, bbox = gis_matrix.reproject_polylines([record], "EPSG:4326", "EPSG:5186")

        self.assertEqual(transformed[0].start_height, 12.5)
        self.assertEqual(transformed[0].end_height, 11.0)
        self.assertIsNotNone(transformed[0].slope_reference_parts)
        self.assertNotEqual(transformed[0].parts[0][0], record.parts[0][0])
        self.assertLess(bbox[0], bbox[2])
        self.assertLess(bbox[1], bbox[3])

    def test_project_layer_keeps_source_geometry_and_uses_working_copy(self) -> None:
        source_crs = crs_support.parse_crs("EPSG:4326")
        project_crs = crs_support.parse_crs("EPSG:5186")
        source_geometry = [
            gis_matrix.GISBoundary(
                rings=[[(126.97, 37.55), (126.99, 37.55), (126.99, 37.57), (126.97, 37.55)]],
                attributes={"NAME": "test"},
            )
        ]
        layer = {
            "kind": "boundary",
            "geometry": source_geometry,
            "bbox": (126.97, 37.55, 126.99, 37.57),
            "meta": source_crs.to_meta(),
        }

        prepared = gis_project.prepare_layer_for_project(layer, project_crs)

        self.assertIs(prepared["source_geometry"], source_geometry)
        self.assertIsNot(prepared["geometry"], source_geometry)
        self.assertTrue(prepared["crs_ready"])
        self.assertEqual(prepared["crs_status"], "reprojected")
        self.assertEqual(prepared["geometry"][0].attributes["NAME"], "test")
        self.assertEqual(prepared["meta"]["project_crs_label"], "EPSG:5186")

    def test_missing_layer_crs_is_not_overlaid_on_defined_project(self) -> None:
        layer = {
            "kind": "polyline",
            "geometry": [gis_matrix.GISPolyline(parts=[[(0.0, 0.0), (1.0, 1.0)]])],
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "meta": {"crs_error": "", "crs_wkt": ""},
        }

        prepared = gis_project.prepare_layer_for_project(layer, crs_support.parse_crs("EPSG:5186"))

        self.assertFalse(prepared["crs_ready"])
        self.assertEqual(prepared["crs_status"], "unassigned")
        self.assertEqual(gis_project.layer_crs_display(prepared), "선택 필요")

    def test_manual_source_assignment_reprojects_without_mutating_source(self) -> None:
        geometry = [gis_matrix.GISPolyline(parts=[[(126.98, 37.56), (126.99, 37.55)]])]
        layer = {
            "kind": "polyline",
            "geometry": geometry,
            "bbox": (126.98, 37.55, 126.99, 37.56),
            "meta": {},
        }

        assigned = gis_project.assign_layer_source_crs(
            layer,
            crs_support.parse_crs("EPSG:4326"),
            crs_support.parse_crs("EPSG:5186"),
        )

        self.assertIs(assigned["source_geometry"], geometry)
        self.assertEqual(geometry[0].parts[0][0], (126.98, 37.56))
        self.assertTrue(assigned["crs_ready"])
        self.assertEqual(assigned["source_meta"]["crs_source"], "manual")

    def test_first_manual_assignment_establishes_project_without_silently_overlaying_other_unknowns(self) -> None:
        selected = {
            "kind": "polyline",
            "geometry": [gis_matrix.GISPolyline(parts=[[(200000.0, 450000.0), (200100.0, 450100.0)]])],
            "bbox": (200000.0, 450000.0, 200100.0, 450100.0),
            "meta": {},
        }
        other = {
            "kind": "boundary",
            "geometry": [
                gis_matrix.GISBoundary(
                    rings=[[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]],
                    attributes={},
                )
            ],
            "bbox": (0.0, 0.0, 1.0, 1.0),
            "meta": {},
        }
        definition = crs_support.parse_crs("EPSG:5186")
        app = object.__new__(gui.NFMATApp)
        app._gis_crs_result_queue = queue.Queue()
        payload = {
            "layer_id": "selected",
            "layer": selected,
            "source_crs": definition,
            "project_crs": definition,
            "establish_project": True,
            "layers": [("selected", selected), ("other", other)],
        }

        app._gis_crs_worker(4, "layer", payload)
        generation, action, result = app._gis_crs_result_queue.get_nowait()

        self.assertEqual((generation, action), (4, "layer"))
        self.assertIsInstance(result, dict)
        prepared = dict(result["layers"])
        self.assertTrue(prepared["selected"]["crs_ready"])
        self.assertFalse(prepared["other"]["crs_ready"])

    def test_missing_pyproj_in_layer_choice_is_reported_without_tk_callback_traceback(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {
            "layer": {
                "name": "missing-prj.shp",
                "kind": "boundary",
                "bbox": (940_000.0, 1_930_000.0, 970_000.0, 1_970_000.0),
                "source_bbox": (940_000.0, 1_930_000.0, 970_000.0, 1_970_000.0),
                "meta": {},
            }
        }
        app._gis_load_in_progress = False
        app._gis_crs_job_polling = False
        app._gis_project_crs = None
        app._selected_gis_layer_id = lambda: "layer"
        app._show_crs_dependency_error = mock.Mock()
        dependency_error = crs_support.CRSDependencyError("pyproj missing")

        with mock.patch("gui.build_layer_crs_choices", side_effect=dependency_error):
            app.configure_selected_layer_crs()

        app._show_crs_dependency_error.assert_called_once_with(dependency_error)

    def test_missing_pyproj_in_project_choice_is_reported_without_tk_callback_traceback(self) -> None:
        app = object.__new__(gui.NFMATApp)
        app._gis_layers = {}
        app._gis_pipe_layer_id = None
        app._gis_load_in_progress = False
        app._gis_crs_job_polling = False
        app._gis_project_crs = None
        app._show_crs_dependency_error = mock.Mock()
        dependency_error = crs_support.CRSDependencyError("pyproj missing")

        with mock.patch("gui.build_project_crs_choices", side_effect=dependency_error):
            app.configure_project_crs()

        app._show_crs_dependency_error.assert_called_once_with(dependency_error)


if __name__ == "__main__":
    unittest.main()
