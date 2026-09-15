import unittest

import crs_catalog
from crs_support import parse_crs


class CRSCatalogTests(unittest.TestCase):
    def test_standard_choices_are_plain_korean_and_parseable(self) -> None:
        choices = crs_catalog.standard_crs_choices()

        self.assertEqual(choices[0].key, "epsg5179")
        self.assertIn("국가 통합 좌표", choices[0].title)
        self.assertTrue(all(choice.definition is not None for choice in choices))

    def test_prj_definition_is_confirmed_without_heuristic_override(self) -> None:
        detected = parse_crs("EPSG:5174")
        choices = crs_catalog.build_layer_crs_choices(detected, parse_crs("EPSG:5186"))

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            (180_000.0, 430_000.0, 220_000.0, 470_000.0),
            detected_crs=detected,
            project_crs=parse_crs("EPSG:5186"),
            project_bbox=(180_000.0, 430_000.0, 220_000.0, 470_000.0),
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "detected")
        self.assertEqual(recommendation.confidence, "confirmed")

    def test_manual_assignment_is_kept_as_a_distinct_user_choice(self) -> None:
        assigned = parse_crs("EPSG:5181")
        choices = crs_catalog.build_layer_crs_choices(None, parse_crs("EPSG:5174"), assigned)

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            (180_000.0, 430_000.0, 220_000.0, 470_000.0),
            assigned_crs=assigned,
        )

        self.assertEqual(choices[0].key, "assigned")
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "assigned")

    def test_realistic_2023_seoul_range_recommends_5179_but_never_claims_confirmation(self) -> None:
        project = parse_crs("EPSG:5174")
        choices = crs_catalog.build_layer_crs_choices(None, project)

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            (940_966.964, 1_939_102.335, 968_819.731, 1_964_112.765),
            layer_name="2023 침수흔적도 최신수정.shp",
            project_crs=project,
            project_bbox=(185_000.0, 438_500.0, 213_000.0, 463_800.0),
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "epsg5179")
        self.assertIn(recommendation.confidence, {"low", "medium"})
        self.assertNotEqual(recommendation.confidence, "confirmed")
        self.assertIn("epsg5178", recommendation.alternatives)
        self.assertIn("자료 제공기관", recommendation.reason)

    def test_same_raw_range_prefers_current_map_choice_without_auto_confirmation(self) -> None:
        project = parse_crs("EPSG:5174")
        bbox = (185_000.0, 438_500.0, 213_000.0, 463_800.0)
        choices = crs_catalog.build_layer_crs_choices(None, project)

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            bbox,
            project_crs=project,
            project_bbox=bbox,
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "project")
        self.assertNotEqual(recommendation.confidence, "confirmed")

    def test_longitude_latitude_range_recommends_gps_format(self) -> None:
        choices = crs_catalog.build_layer_crs_choices(None, None)

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            (126.8, 37.3, 127.2, 37.8),
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "epsg4326")
        self.assertEqual(recommendation.confidence, "medium")

    def test_unknown_numeric_range_is_not_given_a_fake_recommendation(self) -> None:
        choices = crs_catalog.build_layer_crs_choices(None, None)

        recommendation = crs_catalog.recommend_layer_crs(
            choices,
            (50_000_000.0, 70_000_000.0, 50_001_000.0, 70_001_000.0),
        )

        self.assertIsNone(recommendation)

    def test_project_choice_prefers_projected_pipe_coordinates(self) -> None:
        pipe = parse_crs("EPSG:5174")
        choices = crs_catalog.build_project_crs_choices(None, pipe)

        recommendation = crs_catalog.recommend_project_crs(choices, None, pipe)

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.key, "pipe")
        self.assertEqual(recommendation.confidence, "high")


if __name__ == "__main__":
    unittest.main()
