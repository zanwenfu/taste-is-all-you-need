from string_utils import snake_case, truncate, upper


def test_upper():
    assert upper("hi") == "HI"


def test_snake_case():
    assert snake_case("FooBar") == "foo_bar"
    assert snake_case("HTTPServer") == "h_t_t_p_server"


def test_truncate():
    assert truncate("hello", 10) == "hello"
    assert truncate("hello world", 6) == "hello…"
