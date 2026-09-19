"""T321 skeleton smoke: wordstat client seam exists and is deterministic."""
from idea_finder.wordstat.client import FakeWordstatClient, WordstatClient, mask_token


def test_fake_client_is_deterministic() -> None:
    client: WordstatClient = FakeWordstatClient()
    assert client.frequency("не работает принтер") == client.frequency("не работает принтер")


def test_mask_token_hides_short_secrets() -> None:
    assert mask_token("") == "****"
    assert mask_token("abc") == "****"
    assert mask_token("abcdefgh") == "****efgh"
