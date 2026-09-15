import tempfile
import unittest
from pathlib import Path

import numpy as np

import plena_plots


class PlenaPlotTests(unittest.TestCase):
    def test_width_csv_groups_header_beta_runs_and_matches_matlab_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt_width_functions.csv"
            path.write_text(
                "\n".join(
                    [
                        "distance,original,10^-1_run1,10^-1_run2,10^0_run1,10^0_run2",
                        "0,3,1,5,2,8",
                        "1,4,3,7,4,10",
                    ]
                ),
                encoding="utf-8",
            )

            blocks = plena_plots.load_width_function_blocks(path)

        self.assertEqual([block.k for block in blocks], [-1, 0])
        self.assertEqual([block.run_count for block in blocks], [2, 2])
        np.testing.assert_allclose(blocks[0].mean, [3.0, 5.0])
        np.testing.assert_allclose(blocks[0].minimum, [1.0, 3.0])
        np.testing.assert_allclose(blocks[0].maximum, [5.0, 7.0])
        np.testing.assert_allclose(blocks[1].mean, [5.0, 7.0])

    def test_nse_parser_uses_nse_mean_q_column(self) -> None:
        text = "\n".join(
            [
                "[Summary by beta]",
                "k beta ok/total meanNSE(run) NSE(mean_q)",
                "-4 1.0000000000e-04 100/100 0.1981717424 0.7831204982",
                "-3 1.0000000000e-03 100/100 0.2488973127 0.7923741098",
            ]
        )

        points = plena_plots.parse_nse_points_from_text(text)

        self.assertEqual([point.k for point in points], [-4, -3])
        self.assertAlmostEqual(points[0].mean_nse_run, 0.1981717424)
        self.assertAlmostEqual(points[0].nse_mean_q, 0.7831204982)

    def test_plot_renderers_produce_nonblank_images(self) -> None:
        block = plena_plots.WidthFunctionBlock(
            k=-4,
            label="10^-4",
            xi=np.asarray([0.0, 1.0, 2.0]),
            original=np.asarray([0.0, 2.0, 1.0]),
            mean=np.asarray([0.0, 1.0, 3.0]),
            minimum=np.asarray([0.0, 0.5, 2.0]),
            maximum=np.asarray([0.2, 2.0, 4.0]),
            run_count=2,
        )
        nse = [
            plena_plots.NSEPoint(-4, 1e-4, "100/100", 0.2, 0.7),
            plena_plots.NSEPoint(-3, 1e-3, "100/100", 0.3, 0.8),
        ]

        width_image = plena_plots.render_width_function_plot(block, size=(500, 360))
        nse_image = plena_plots.render_nse_plot(nse, size=(420, 360))

        self.assertEqual(width_image.size, (500, 360))
        self.assertEqual(nse_image.size, (420, 360))
        self.assertGreater(int(np.count_nonzero(np.asarray(width_image) != 255)), 1000)
        self.assertGreater(int(np.count_nonzero(np.asarray(nse_image) != 255)), 1000)


if __name__ == "__main__":
    unittest.main()
