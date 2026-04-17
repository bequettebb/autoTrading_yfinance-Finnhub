"""Write human-readable and machine-readable reports for the paper-trading bot."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt_float(value: float | None, digits: int = 2) -> str:
    """Format optional floats consistently for markdown reports."""

    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def write_daily_report(
    *,
    report_root: str | Path,
    generated_at: datetime,
    trading_mode: str,
    strategy_name: str,
    market_open: bool,
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    leader_symbol: str | None,
    notes: list[str],
    news_analysis: dict[str, Any] | None = None,
) -> Path:
    """Write the latest daily markdown snapshot and a latest pointer file."""

    local_time = generated_at.astimezone()
    report_dir = Path(report_root) / "daily"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{local_time.date().isoformat()}.md"

    lines: list[str] = []
    lines.append(f"# Trading Report {local_time.date().isoformat()}")
    lines.append("")
    lines.append(f"Generated: {local_time.isoformat()}")
    lines.append(f"Mode: {trading_mode}")
    lines.append(f"Strategy: {strategy_name}")
    lines.append(f"Market Open: {'yes' if market_open else 'no'}")
    lines.append(f"Leader: {leader_symbol or 'none'}")
    lines.append("")
    lines.append("## Account")
    lines.append("")
    lines.append(f"- Equity: {_fmt_float(account.get('equity'))}")
    lines.append(f"- Cash: {_fmt_float(account.get('cash'))}")
    lines.append(f"- Exposure: {_fmt_float(account.get('current_exposure'))}")
    if "daily_pnl" in account:
        lines.append(
            f"- Daily P/L: {_fmt_float(account.get('daily_pnl'))} ({_fmt_float(account.get('daily_pnl_pct'), 2)}%)"
        )
    lines.append(f"- Trading Blocked: {account.get('trading_blocked')}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    if notes:
        for note in notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No cycle notes were recorded.")
    if news_analysis is not None and news_analysis.get("summary"):
        lines.append(f"- News overlay: {news_analysis.get('summary')}")
    lines.append("")
    lines.append("## Open Positions")
    lines.append("")
    if positions:
        for position in positions:
            lines.append(
                f"- {position['symbol']}: qty={_fmt_float(position['qty'], 4)} "
                f"avg={_fmt_float(position['avg_entry_price'])} "
                f"value={_fmt_float(position['market_value'])}"
            )
    else:
        lines.append("- No open positions.")
    lines.append("")
    lines.append("## Evaluations")
    lines.append("")
    if evaluations:
        for item in evaluations:
            lines.append(
                f"- {item['symbol']}: action={item['action']} score={_fmt_float(item['score'], 5)} "
                f"{item.get('metric_bias_label', 'Momentum')}={_fmt_float(item.get('metric_bias_value'), 4)} "
                f"{item.get('metric_primary_label', 'Fast')}={_fmt_float(item.get('metric_primary_value'))} "
                f"{item.get('metric_secondary_label', 'Slow')}={_fmt_float(item.get('metric_secondary_value'))} "
                f"last={_fmt_float(item['last_close'])}"
            )
            if item.get("reason"):
                lines.append(f"  reason: {item.get('reason')}")
            if item.get("llm_reason"):
                lines.append(f"  news: {item.get('llm_reason')}")
    else:
        lines.append("- No symbol evaluations were available in the latest cycle.")
    lines.append("")
    lines.append("## News Overlay")
    lines.append("")
    if news_analysis is None:
        lines.append("- News reasoning was not available in the latest cycle.")
    else:
        lines.append(f"- Status: {news_analysis.get('status', '-')}")
        lines.append(f"- Provider: {news_analysis.get('provider', '-')}")
        if news_analysis.get("model"):
            lines.append(f"- Model: {news_analysis.get('model')}")
        if news_analysis.get("overall_sentiment"):
            lines.append(f"- Overall Sentiment: {news_analysis.get('overall_sentiment')}")
        if news_analysis.get("confidence") is not None:
            lines.append(f"- Confidence: {_fmt_float(news_analysis.get('confidence'), 2)}")
        if news_analysis.get("summary"):
            lines.append(f"- Summary: {news_analysis.get('summary')}")
        if news_analysis.get("leader_explanation"):
            lines.append(f"- Leader Context: {news_analysis.get('leader_explanation')}")
        warnings = news_analysis.get("warnings", [])
        if isinstance(warnings, list) and warnings:
            for warning in warnings:
                lines.append(f"- Warning: {warning}")
        article_briefs = news_analysis.get("article_briefs", [])
        if isinstance(article_briefs, list) and article_briefs:
            lines.append("- Key Article Briefs:")
            for article in article_briefs[:5]:
                if not isinstance(article, dict):
                    continue
                source = article.get("source", "Unknown")
                headline = article.get("headline", "-")
                updated_at = article.get("updated_at", "-")
                summary = article.get("summary", "")
                lines.append(f"  - {updated_at} | {source} | {headline}")
                if summary:
                    lines.append(f"    {summary}")
        else:
            lines.append("- No recent Finnhub articles matched the watched symbols.")
    lines.append("")
    lines.append("## Recent Orders")
    lines.append("")
    if orders:
        for order in orders:
            submitted_at = order.get("submitted_at", "-")
            lines.append(
                f"- {submitted_at} | {order['symbol']} | {order['side']} | status={order['status']} "
                f"filled_qty={order.get('filled_qty', '-')} filled_avg={order.get('filled_avg_price', '-')}"
            )
    else:
        lines.append("- No recent orders found.")
    lines.append("")

    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    latest_path = Path(report_root) / "latest.md"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(report_text, encoding="utf-8")
    return report_path


def write_status_snapshot(
    *,
    report_root: str | Path,
    snapshot: dict[str, Any],
) -> Path:
    """Write the latest machine-readable bot snapshot used by the local dashboard."""

    report_root_path = Path(report_root)
    report_root_path.mkdir(parents=True, exist_ok=True)
    status_path = report_root_path / "status.json"
    status_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return status_path
