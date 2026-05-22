# Security Policy

## Supported versions

This project is a lightweight agentic skill and CLI. Security fixes are handled on the `main` branch.

## Reporting a vulnerability

Please do **not** open a public issue for vulnerabilities involving secrets, credential leakage, unsafe file handling, or API misuse.

Report privately through GitHub's security advisory flow if available, or contact the repository owner through their public GitHub profile.

When reporting, include:

- A short summary of the issue.
- Reproduction steps or a minimal proof of concept.
- Impact and affected files.
- Any suggested fix, if known.

## Secret handling

- Never commit `OPENAI_API_KEY`, `.env`, or other credentials.
- The CLI reads `OPENAI_API_KEY` / `OPENAI_BASE_URL`, or `ANTHROPIC_AUTH_TOKEN` / `ANTHROPIC_BASE_URL`, from the current directory `.env`, then `~/.env`, then process env.
- Generated examples and issue reports should not include private prompts, private images, or API response bodies containing secrets.
