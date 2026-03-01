"""
Unit tests untuk fitur generate_topic_summary di app.py.

Yang ditest:
- Sampling konten: limit chars, prioritas negatif saat high risk
- Handling edge cases: empty df, no content, no API key
- Output format: return tuple (summary, error)
- HIDDEN_TOPICS filtering
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────
# Import function under test
# ─────────────────────────────────────────────
from app import generate_topic_summary, HIDDEN_TOPICS


# ─────────────────────────────────────────────
# Helper: buat dummy DataFrame topik
# ─────────────────────────────────────────────
def make_topic_df(n=20, sentiment_dist=None):
    """Buat DataFrame mirip df_filtered yang sudah difilter per topik."""
    if sentiment_dist is None:
        sentiment_dist = ['negative'] * 10 + ['positive'] * 7 + ['neutral'] * 3
    contents = [f"Konten diskusi masyarakat tentang kebijakan nomor {i}" for i in range(n)]
    return pd.DataFrame({
        'content': contents[:n],
        'final_sentiment': (sentiment_dist * ((n // len(sentiment_dist)) + 1))[:n],
        'final_topic': ['Test Topic'] * n,
        'date': pd.date_range('2024-01-01', periods=n, freq='D'),
    })


def make_empty_topic_df():
    return pd.DataFrame({
        'content': pd.Series([], dtype='object'),
        'final_sentiment': pd.Series([], dtype='object'),
        'final_topic': pd.Series([], dtype='object'),
        'date': pd.Series([], dtype='datetime64[ns]'),
    })


# ─────────────────────────────────────────────
# TEST: No API key → returns error
# ─────────────────────────────────────────────
class TestNoApiKey:
    @patch('app.get_openai_client', return_value=None)
    def test_returns_error_when_no_api_key(self, mock_client):
        df = make_topic_df()
        summary, error = generate_topic_summary(df, "Test Topic", 50.0, "High")
        assert summary is None
        assert "API key" in error


# ─────────────────────────────────────────────
# TEST: Empty / no content
# ─────────────────────────────────────────────
class TestEmptyContent:
    @patch('app.get_openai_client')
    def test_empty_dataframe(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        df = make_empty_topic_df()
        summary, error = generate_topic_summary(df, "Test", 0.0, "Low")
        assert summary is None
        assert "No content" in error

    @patch('app.get_openai_client')
    def test_all_nan_content(self, mock_get_client):
        mock_get_client.return_value = MagicMock()
        df = pd.DataFrame({
            'content': [np.nan, np.nan, None],
            'final_sentiment': ['positive', 'negative', 'neutral'],
        })
        summary, error = generate_topic_summary(df, "Test", 0.0, "Low")
        assert summary is None
        assert "No content" in error


# ─────────────────────────────────────────────
# TEST: Content sampling limits
# ─────────────────────────────────────────────
class TestContentSampling:
    @patch('app.get_openai_client')
    def test_total_chars_under_limit(self, mock_get_client):
        """Sampled content harus <= 4500 chars."""
        # Buat konten panjang per item
        long_content = "A" * 500
        df = pd.DataFrame({
            'content': [long_content] * 50,
            'final_sentiment': ['negative'] * 50,
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        stream, error = generate_topic_summary(df, "Test", 60.0, "High", api_key="test")

        # Cek bahwa API dipanggil
        assert mock_client.chat.completions.create.called

        # Cek prompt yang dikirim ke API
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']

        # Hitung jumlah konten yang di-sample (dipisah oleh \n---\n)
        content_parts = user_msg.split("---")
        # Setiap 500 char content → max 4500/500 = 9 items
        assert len(content_parts) <= 10

    @patch('app.get_openai_client')
    def test_per_content_max_500_chars(self, mock_get_client):
        """Setiap konten individual di-truncate ke 500 chars."""
        very_long = "B" * 1000  # 1000 chars
        df = pd.DataFrame({
            'content': [very_long] * 5,
            'final_sentiment': ['positive'] * 5,
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']

        # Seharusnya tidak ada substring 1000 B berturut-turut
        assert "B" * 501 not in user_msg


# ─────────────────────────────────────────────
# TEST: High risk oversamples negative content
# ─────────────────────────────────────────────
class TestHighRiskSampling:
    @patch('app.get_openai_client')
    def test_high_risk_prioritizes_negative(self, mock_get_client):
        """Saat risk High, pool harus berisi lebih banyak konten negatif."""
        neg_contents = [f"Konten negatif {i}" for i in range(30)]
        pos_contents = [f"Konten positif {i}" for i in range(30)]

        df = pd.DataFrame({
            'content': neg_contents + pos_contents,
            'final_sentiment': ['negative'] * 30 + ['positive'] * 30,
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 70.0, "High", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']

        # Pool: 20 negative + 10 other → negative harus dominan
        neg_count = user_msg.count("Konten negatif")
        pos_count = user_msg.count("Konten positif")
        assert neg_count >= pos_count


# ─────────────────────────────────────────────
# TEST: Successful API call returns summary
# ─────────────────────────────────────────────
class TestSuccessfulCall:
    @patch('app.get_openai_client')
    def test_returns_summary_on_success(self, mock_get_client):
        df = make_topic_df(10)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "**Ringkasan Topik**: Ini adalah ringkasan."
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        summary, error = generate_topic_summary(df, "Test", 45.0, "Moderate", api_key="test")
        assert summary == "**Ringkasan Topik**: Ini adalah ringkasan."
        assert error is None

    @patch('app.get_openai_client')
    def test_uses_gpt_4_1_mini_model(self, mock_get_client):
        """Harus menggunakan model gpt-4.1-mini."""
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4.1-mini'

    @patch('app.get_openai_client')
    def test_max_tokens_is_600(self, mock_get_client):
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['max_tokens'] == 600


# ─────────────────────────────────────────────
# TEST: API error handling
# ─────────────────────────────────────────────
class TestApiError:
    @patch('app.get_openai_client')
    def test_api_exception_returns_error(self, mock_get_client):
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API rate limit")
        mock_get_client.return_value = mock_client

        summary, error = generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")
        assert summary is None
        assert "API rate limit" in error


# ─────────────────────────────────────────────
# TEST: HIDDEN_TOPICS constant
# ─────────────────────────────────────────────
class TestHiddenTopics:
    def test_contains_other_topic(self):
        assert 'other-topic' in HIDDEN_TOPICS

    def test_contains_onm_unclustered(self):
        assert 'tidak ada topic / cluster -1 (unclustered)' in HIDDEN_TOPICS

    def test_is_a_set(self):
        assert isinstance(HIDDEN_TOPICS, set)


# ─────────────────────────────────────────────
# TEST: Prompt format
# ─────────────────────────────────────────────
class TestPromptFormat:
    @patch('app.get_openai_client')
    def test_prompt_contains_topic_name(self, mock_get_client):
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Kebijakan Suku Bunga", 30.0, "Moderate", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']
        assert "Kebijakan Suku Bunga" in user_msg

    @patch('app.get_openai_client')
    def test_prompt_contains_risk_level(self, mock_get_client):
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 60.0, "High", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']
        assert "High" in user_msg
        assert "Ringkasan Topik" in user_msg
        assert "Isu Utama" in user_msg
        assert "Sentimen Publik" in user_msg

    @patch('app.get_openai_client')
    def test_system_message_is_indonesian(self, mock_get_client):
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")

        call_args = mock_client.chat.completions.create.call_args
        system_msg = call_args[1]['messages'][0]['content']
        assert "Bank Indonesia" in system_msg


# ─────────────────────────────────────────────
# TEST: Risk data included in prompt
# ─────────────────────────────────────────────
class TestRiskDataInPrompt:
    @patch('app.get_openai_client')
    def test_risk_data_included_when_provided(self, mock_get_client):
        """Saat risk_data dikirim, prompt harus mengandung data risk score."""
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        risk_data = {
            'risk_score': 0.72,
            'negative_ratio': 0.65,
            'velocity': 0.30,
            'influencer_impact': 0.80,
            'misinformation_score': 0.25,
            'total_data': 150,
        }

        generate_topic_summary(df, "Test", 65.0, "High", api_key="test", risk_data=risk_data)

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']

        # Prompt harus mengandung semua komponen risk score
        assert "Risk Score: 0.72" in user_msg
        assert "Negative Ratio: 0.65" in user_msg
        assert "Velocity" in user_msg
        assert "Influencer Impact: 0.80" in user_msg
        assert "Misinformation Score: 0.25" in user_msg
        assert "150" in user_msg

    @patch('app.get_openai_client')
    def test_no_risk_data_still_works(self, mock_get_client):
        """Tanpa risk_data, prompt tetap jalan tanpa crash."""
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Summary without risk"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        summary, error = generate_topic_summary(df, "Test", 10.0, "Low", api_key="test")
        assert summary == "Summary without risk"
        assert error is None

        # Prompt tidak mengandung "Risk Score:" karena risk_data=None
        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']
        assert "Risk Score:" not in user_msg

    @patch('app.get_openai_client')
    def test_prompt_references_risk_score_in_explanation(self, mock_get_client):
        """Prompt harus minta LLM referensikan angka risk score saat menjelaskan risiko."""
        df = make_topic_df(5)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock()
        mock_get_client.return_value = mock_client

        risk_data = {
            'risk_score': 0.45,
            'negative_ratio': 0.35,
            'velocity': 0.20,
            'influencer_impact': 0.50,
            'misinformation_score': 0.10,
            'total_data': 80,
        }

        generate_topic_summary(df, "Test", 35.0, "Moderate", api_key="test", risk_data=risk_data)

        call_args = mock_client.chat.completions.create.call_args
        user_msg = call_args[1]['messages'][1]['content']

        # Prompt harus instruksikan LLM untuk referensikan data risk score
        assert "risk score" in user_msg.lower()
