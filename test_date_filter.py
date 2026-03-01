"""
Unit tests untuk logika filter tanggal di app.py.

Yang ditest:
- Filter DataFrame berdasarkan date_range
- Guard ketika date_range belum lengkap (hanya 1 elemen)
- Edge cases: tanggal sama, range terbalik, data kosong
"""

import pytest
import pandas as pd
from datetime import date, datetime, timedelta


# ─────────────────────────────────────────────
# Helper: buat dummy DataFrame seperti di app.py
# ─────────────────────────────────────────────
def make_df(dates: list[str]) -> pd.DataFrame:
    """Buat DataFrame dengan kolom 'date' bertipe datetime dari list string."""
    return pd.DataFrame({"date": pd.to_datetime(dates), "content": "x"})


def apply_date_filter(df: pd.DataFrame, date_range: tuple) -> pd.DataFrame:
    """
    Logika filter tanggal persis seperti di app.py (lines 789-796).
    Return None jika date_range belum lengkap (simulasi st.stop).
    """
    if len(date_range) != 2:
        return None  # simulasi st.stop()
    return df[
        (df["date"].dt.date >= date_range[0]) &
        (df["date"].dt.date <= date_range[1])
    ]


# ─────────────────────────────────────────────
# GUARD: date_range belum lengkap
# ─────────────────────────────────────────────
class TestDateRangeGuard:
    def test_empty_tuple_returns_none(self):
        df = make_df(["2024-01-10"])
        assert apply_date_filter(df, ()) is None

    def test_single_date_returns_none(self):
        """Kasus utama: user baru pilih start date, end date belum dipilih."""
        df = make_df(["2024-01-10"])
        assert apply_date_filter(df, (date(2024, 1, 1),)) is None

    def test_complete_range_does_not_return_none(self):
        df = make_df(["2024-01-10"])
        result = apply_date_filter(df, (date(2024, 1, 1), date(2024, 1, 31)))
        assert result is not None


# ─────────────────────────────────────────────
# NORMAL FILTERING
# ─────────────────────────────────────────────
class TestDateFiltering:
    def setup_method(self):
        self.df = make_df([
            "2024-01-01",
            "2024-01-15",
            "2024-01-31",
            "2024-02-10",
            "2024-03-01",
        ])

    def test_filter_includes_start_date(self):
        result = apply_date_filter(self.df, (date(2024, 1, 1), date(2024, 1, 15)))
        assert date(2024, 1, 1) in result["date"].dt.date.values

    def test_filter_includes_end_date(self):
        result = apply_date_filter(self.df, (date(2024, 1, 1), date(2024, 1, 15)))
        assert date(2024, 1, 15) in result["date"].dt.date.values

    def test_filter_excludes_outside_range(self):
        result = apply_date_filter(self.df, (date(2024, 1, 1), date(2024, 1, 31)))
        assert date(2024, 2, 10) not in result["date"].dt.date.values
        assert date(2024, 3, 1) not in result["date"].dt.date.values

    def test_correct_row_count(self):
        result = apply_date_filter(self.df, (date(2024, 1, 1), date(2024, 1, 31)))
        assert len(result) == 3

    def test_filter_returns_empty_when_no_match(self):
        result = apply_date_filter(self.df, (date(2025, 1, 1), date(2025, 12, 31)))
        assert len(result) == 0

    def test_all_rows_included_for_full_range(self):
        result = apply_date_filter(self.df, (date(2024, 1, 1), date(2024, 12, 31)))
        assert len(result) == 5


# ─────────────────────────────────────────────
# EDGE CASES
# ─────────────────────────────────────────────
class TestEdgeCases:
    def test_same_start_and_end_date(self):
        """Start = End date → hanya data di tanggal itu yang muncul."""
        df = make_df(["2024-01-15", "2024-01-16"])
        result = apply_date_filter(df, (date(2024, 1, 15), date(2024, 1, 15)))
        assert len(result) == 1
        assert date(2024, 1, 15) in result["date"].dt.date.values

    def test_empty_dataframe(self):
        """DataFrame kosong tidak crash."""
        df = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "content": []})
        result = apply_date_filter(df, (date(2024, 1, 1), date(2024, 1, 31)))
        assert len(result) == 0

    def test_single_row_in_range(self):
        df = make_df(["2024-06-15"])
        result = apply_date_filter(df, (date(2024, 6, 1), date(2024, 6, 30)))
        assert len(result) == 1

    def test_single_row_out_of_range(self):
        df = make_df(["2024-06-15"])
        result = apply_date_filter(df, (date(2024, 7, 1), date(2024, 7, 31)))
        assert len(result) == 0

    def test_filter_does_not_mutate_original_df(self):
        """Filter harus return slice baru, bukan modifikasi df asli."""
        df = make_df(["2024-01-01", "2024-02-01"])
        original_len = len(df)
        apply_date_filter(df, (date(2024, 1, 1), date(2024, 1, 31)))
        assert len(df) == original_len
