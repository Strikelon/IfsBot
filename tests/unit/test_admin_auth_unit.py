import bcrypt

from admin.auth import issue_session_token, verify_password, verify_session_token

_SECRET = "test-secret"


def test_verify_password():
    hashed = bcrypt.hashpw(b"hunter2", bcrypt.gensalt()).decode()
    assert verify_password("hunter2", hashed) is True
    assert verify_password("wrong", hashed) is False


def test_verify_password_handles_bad_hash():
    assert verify_password("x", "not-a-bcrypt-hash") is False


def test_session_token_roundtrip():
    token = issue_session_token("boss", secret=_SECRET)
    assert verify_session_token(token, secret=_SECRET, max_age_days=7) == "boss"


def test_expired_or_tampered_token_rejected():
    token = issue_session_token("boss", secret=_SECRET)
    assert verify_session_token(token + "x", secret=_SECRET, max_age_days=7) is None
    assert verify_session_token(token, secret="other", max_age_days=7) is None
