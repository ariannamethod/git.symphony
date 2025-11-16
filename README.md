# repoview

> Just show me the code.

Flatten any GitHub repository into a single, static HTML page with syntax highlighting and instant search.

## Install

```bash
uv tool install git+https://github.com/karpathy/rendergit
```

Or with pip:

```bash
pip install git+https://github.com/karpathy/rendergit
```

## Usage

```bash
repoview https://github.com/karpathy/nanoGPT
```

This will:
1. Clone the repo to a temp directory
2. Generate a single HTML file with all source code
3. Open it in your browser

## Features

**👤 Human View**
- Syntax highlighting via Pygments
- Markdown rendering for docs
- Sidebar navigation
- Directory tree overview
- Smart filtering (skips binaries and large files)
- Instant search with Ctrl+F

**🤖 LLM View**
- One-click toggle to CXML format
- Copy entire codebase to paste into Claude, ChatGPT, etc.
- Optimized for AI code analysis

## Options

```bash
repoview <repo-url> [--out file.html] [--max-bytes N] [--no-open]
```

## License

0BSD
