# Contributing

Taste is research software by Zanwen Fu. Bug reports, counter-examples and
failure cases are the most useful contributions: open an
[issue](https://github.com/zanwenfu/taste-is-all-you-need/issues) with what
you ran and what happened.

## Working on the code

```bash
pip install -e '.[dev,brains,openai]'
ruff check taste/ tests/ examples/
pytest
```

No test needs an API key or a network. CI runs the same three commands on
Python 3.11, 3.12 and 3.14, then the hermetic demo.

- For anything larger than a small fix, open an issue first so the direction
  can be agreed before you spend time on it.
- A change in behaviour comes with a test that fails without it.
- Keep a subsystem removable: what is off by default must change nothing when
  it is off.

## Licence of contributions

The project is licensed under Apache-2.0 (see [LICENSE](LICENSE) and
[NOTICE](NOTICE)). By submitting a contribution you agree that it is licensed
under the same terms.
