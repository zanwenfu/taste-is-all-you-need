"""Paper 1: what an agent told something long ago still uses, when its memory is Taste's and it may search it.

A rule-loss case gives a long history, the summary that replaced it, and a question the rule should decide. The
history is also committed into Taste's real memory, one turn per commit, and the model answering is offered
search and fetch. The steps are counted apart: stored (by construction), looks, finds, uses.
"""
