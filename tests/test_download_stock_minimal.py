import tempfile
import unittest
from pathlib import Path

from download_stock import STATIC_SYMBOLS, build_merge_config, parse_symbols


class TestDownloadStockMinimal(unittest.TestCase):
    def test_static_symbols_remain_parseable(self):
        symbols = parse_symbols(STATIC_SYMBOLS[:3])

        self.assertEqual(len(symbols), 3)
        self.assertEqual(symbols[0].symbol.count("."), 1)
        self.assertIn(symbols[0].market, {"SH", "SZ"})

    def test_build_merge_config_points_to_requested_parquet_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            parquet_dir = Path(tmp_dir) / "daily"

            config = build_merge_config(parquet_dir)

            self.assertEqual(config.paths.raw_daily_dir, parquet_dir.resolve())
            self.assertEqual(config.paths.merged_parquet.name, "market_daily.parquet")


if __name__ == "__main__":
    unittest.main()
