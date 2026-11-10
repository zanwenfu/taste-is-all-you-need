"""The write-time date pass for a raw history (Nov 9): what it dates, what it leaves alone, and that it dates once."""

import datetime as dt

from studies.memory_bench.dates import dated_session, session_day, stamped_session, write_dates

SATURDAY = dt.date(2023, 5, 20)


def test_each_known_expression_is_followed_by_its_date() -> None:
    cases = {
        "I started it yesterday.": "I started it yesterday [2023-05-19].",
        "We met the day before yesterday": "We met the day before yesterday [2023-05-18]",
        "Done today, due tomorrow": "Done today [2023-05-20], due tomorrow [2023-05-21]",
        "I bought it two weeks ago": "I bought it two weeks ago [2023-05-06]",
        "about 3 days ago": "about 3 days ago [2023-05-17]",
        "a couple of days ago": "a couple of days ago [2023-05-18]",
        "six months ago": "six months ago [2022-11]",
        "a year ago": "a year ago [2022]",
        "in 3 days": "in 3 days [2023-05-23]",
        "in a month": "in a month [2023-06]",
        "last week": "last week [the week of 2023-05-08]",
        "next week": "next week [the week of 2023-05-22]",
        "last month": "last month [2023-04]",
        "next year": "next year [2024]",
        "last weekend": "last weekend [the weekend of 2023-05-13]",
        "last Friday": "last Friday [2023-05-19]",
        "last Saturday": "last Saturday [2023-05-13]",
        "next Monday": "next Monday [2023-05-22]",
        "Yesterday's game": "Yesterday [2023-05-19]'s game",
    }
    for text, dated in cases.items():
        assert write_dates(text, SATURDAY) == dated, text


def test_expressions_that_could_name_two_dates_are_left_alone() -> None:
    for text in ("on Friday", "this Friday", "a few days ago", "some weeks ago", "the 2023 season", "in May", "lastly", "todays"):
        assert write_dates(text, SATURDAY) == text, text


def test_a_text_is_dated_once() -> None:
    text = "the day before yesterday and yesterday, two weeks ago, last Friday"
    once = write_dates(text, SATURDAY)
    assert write_dates(once, SATURDAY) == once
    assert once.count("[") == 4


def test_each_benchmarks_session_date_is_read_and_anything_else_dates_nothing() -> None:
    assert session_day("2023/05/20 (Sat) 02:21") == SATURDAY
    assert session_day("1:56 pm on 8 May, 2023") == dt.date(2023, 5, 8)
    assert session_day("10:37 am on 27 June, 2023") == dt.date(2023, 6, 27)
    assert session_day("sometime") is None
    session = {"sid": "s1", "date": "sometime", "turns": [{"role": "user", "content": "yesterday"}]}
    assert dated_session(session) is session


def test_the_dated_store_holds_and_delivers_dated_text_and_plain_taste_does_not(tmp_path) -> None:
    from studies.memory_bench.run import make_system

    session = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21",
               "turns": [{"role": "user", "content": "I started reading The Nightingale yesterday."}, {"role": "assistant", "content": "Enjoy it."}]}
    dated, plain = make_system("taste_git_dated", None, tmp_path), make_system("taste_git", None, tmp_path)
    try:
        dated.ingest([session])
        plain.ingest([session])
        got, kept = dated.retrieve("Nightingale reading")[0], plain.retrieve("Nightingale reading")[0]
        assert got.turns[0]["content"] == "I started reading The Nightingale yesterday [2023-05-19]."
        assert kept.turns[0]["content"] == "I started reading The Nightingale yesterday."
        assert (got.session, got.rounds, got.date) == (kept.session, kept.rounds, kept.date)
    finally:
        dated.close()
        plain.close()


def test_only_what_people_said_is_dated_not_the_assistants_replies() -> None:
    lme = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21", "turns": [{"role": "user", "content": "I moved in last week."},
                                                                      {"role": "assistant", "content": "How can I help you today?"}]}
    dated = dated_session(lme)
    assert dated["turns"][0]["content"] == "I moved in last week [the week of 2023-05-08]."
    assert dated["turns"][1]["content"] == "How can I help you today?"
    locomo = {"sid": "c1", "date": "1:56 pm on 8 May, 2023", "turns": [{"role": "Caroline", "content": "I went yesterday."},
                                                                         {"role": "Melanie", "content": "Me too, last Friday."}]}
    assert [turn["content"] for turn in dated_session(locomo)["turns"]] == ["I went yesterday [2023-05-07].", "Me too, last Friday [2023-05-05]."]


def test_a_session_keeps_its_id_date_and_roles_and_only_its_text_changes() -> None:
    session = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21", "turns": [{"role": "user", "content": "I finished it yesterday."},
                                                                          {"role": "assistant", "content": "Well done!", "has_answer": True}]}
    dated = dated_session(session)
    assert dated["sid"] == "s1" and dated["date"] == session["date"]
    assert dated["turns"][0] == {"role": "user", "content": "I finished it yesterday [2023-05-19]."}
    assert dated["turns"][1] == session["turns"][1]
    assert session["turns"][0]["content"] == "I finished it yesterday.", "the original session is not changed"


def test_each_persons_turn_is_stamped_with_its_sessions_day_on_top_of_the_date_pass() -> None:
    session = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21", "turns": [{"role": "user", "content": "I moved in last week."},
                                                                          {"role": "user", "content": "I just started reading The Nightingale."},
                                                                          {"role": "assistant", "content": "How can I help you today?"}]}
    stamped = stamped_session(session)
    assert [turn["content"] for turn in stamped["turns"]] == ["[2023-05-20] I moved in last week [the week of 2023-05-08].",
                                                               "[2023-05-20] I just started reading The Nightingale.",
                                                               "How can I help you today?"]
    assert stamped["sid"] == "s1" and stamped["date"] == session["date"]
    assert session["turns"][0]["content"] == "I moved in last week.", "the original session is not changed"


def test_a_session_is_stamped_once_and_an_undated_one_not_at_all() -> None:
    session = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21", "turns": [{"role": "user", "content": "I finished it yesterday."}]}
    once = stamped_session(session)
    assert stamped_session(once) == once
    undated = {"sid": "s2", "date": "sometime", "turns": [{"role": "user", "content": "yesterday"}]}
    assert stamped_session(undated) is undated


def test_both_of_locomos_speakers_are_stamped() -> None:
    locomo = {"sid": "c1", "date": "1:56 pm on 8 May, 2023", "turns": [{"role": "Caroline", "content": "I went yesterday."},
                                                                         {"role": "Melanie", "content": "Me too."}]}
    assert [turn["content"] for turn in stamped_session(locomo)["turns"]] == ["[2023-05-08] I went yesterday [2023-05-07].", "[2023-05-08] Me too."]


def test_the_stamped_store_holds_and_delivers_stamped_text(tmp_path) -> None:
    from studies.memory_bench.run import make_system

    session = {"sid": "s1", "date": "2023/05/20 (Sat) 02:21",
               "turns": [{"role": "user", "content": "I started reading The Nightingale yesterday."}, {"role": "assistant", "content": "Enjoy it."}]}
    stamped, plain = make_system("taste_git_stamped", None, tmp_path), make_system("taste_git", None, tmp_path)
    try:
        stamped.ingest([session])
        plain.ingest([session])
        got, kept = stamped.retrieve("Nightingale reading")[0], plain.retrieve("Nightingale reading")[0]
        assert got.turns[0]["content"] == "[2023-05-20] I started reading The Nightingale yesterday [2023-05-19]."
        assert got.turns[1]["content"] == "Enjoy it."
        assert (got.session, got.rounds, got.date) == (kept.session, kept.rounds, kept.date)
    finally:
        stamped.close()
        plain.close()
