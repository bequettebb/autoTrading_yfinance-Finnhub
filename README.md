# Auto Trading Codex (Finnhub Only version)

Market monitor that uses yfinance for stock prices and Finnhub for news, then renders a real-time dashboard + news context.

## Quick Start (for macOS)

```bash
cd ~your location
cp .env.example .env
```

Set at least:

```env
FINNHUB_API_KEY=your_finnhub_api_key
```

Run:

```bash
python main.py --once
python main.py --loop
```

Background scripts:

```bash
./scripts/start_bot.sh
./scripts/status_bot.sh
./scripts/stop_bot.sh
./scripts/start_dashboard.sh
./scripts/status_dashboard.sh
./scripts/stop_dashboard.sh
```

Convenience launchers:

```text
./start_bot.sh
./status_bot.sh
./stop_bot.sh
./start_dashboard.sh
./status_dashboard.sh
./stop_dashboard.sh
./start_bot_and_dashboard.sh
./open_latest_report.sh
```

## What It Does

- Fetches stock candles from yfinance for `SYMBOLS`
- Computes strategy signals (`leveraged_rotation` or `breakout_momentum`)
- Fetches Finnhub news for SOXX-focused universe
- Builds heuristic and optional Gemini overlays for the dashboard
- Writes:
  - `reports/latest.md`
  - `reports/daily/YYYY-MM-DD.md`
  - `reports/status.json`

## News Settings

```env
NEWS_REASONING_ENABLED=true
NEWS_PROVIDER=finnhub
NEWS_REALTIME_LOOKBACK_MINUTES=90
NEWS_MAX_ARTICLES=8
NEWS_FOCUS_HOLDINGS_COUNT=10
NEWS_ARTICLE_BRIEF_COUNT=4
GEMINI_API_KEY=
```

If `GEMINI_API_KEY` is blank, dashboard still shows heuristic news analysis.
