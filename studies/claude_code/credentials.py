"""Which credential the study runs on, in a module with no other imports: the rule reader imports it inside the task containers, some
of which have Python 3.8, where the driver's own imports fail (seen Nov 16 2026 at the head-to-head's first task)."""

SUBSCRIPTION_TOKEN = "CLAUDE_CODE_OAUTH_TOKEN"  # the user's Claude subscription; read from its file by the caller, never saved
OTHER_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                     "CLAUDE_CODE_USE_MANTLE", "AWS_BEARER_TOKEN_BEDROCK")
