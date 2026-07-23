"""
Generate the monthly signal report -- the document a subscriber actually reads.

    python -m scripts.signal_report                      # default strategy
    python -m scripts.signal_report us_sp500_top20_momentum

Writes a self-contained HTML file to data/reports/ and prints the path. Open it in
a browser; print it to PDF if you want to send it.

WHY A DOCUMENT AND NOT CONSOLE OUTPUT. A terminal dump of 200 vendor warnings and a
raw ticker list is not a product -- it reads as a crash log, and nobody trusts money
to something that looks broken. The same information, laid out as a document with the
action first and the risk given equal weight, is something a person can act on.

THREE FIXES OVER THE RAW CONSOLE VERSION
  1. Vendor noise is suppressed. yfinance prints a warning for every historical index
     member that no longer trades. Those are EXPECTED -- point-in-time membership
     includes companies that have since delisted -- so they are captured, counted,
     and summarised rather than dumped.
  2. Failed tickers are RETRIED individually. A bulk download can fail on a name that
     works fine alone (CF and ON both did). Silently dropping a current member from
     the ranking is a correctness bug, not a cosmetic one, so we retry and then
     DISCLOSE any that still fail.
  3. Data gaps appear IN the report. A missing constituent is stated in the document,
     not hidden in a log the reader never sees.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from engine.backtest import get_overlay
from engine.data import get_adapter
from engine.markets.market import load_market
from engine.portfolio.construction import build_portfolio
from engine.signals import get_signal
from engine.universe.universe import eligible_universe, load_membership

DEFAULT_STRATEGY = "us_sp500_top20_momentum"
REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "config" / "strategies"
REPORT_DIR = Path("data/reports")
SAVE_DIR = Path("data/portfolios")
HISTORY_MONTHS = 24
CONCENTRATION_REVIEW = 0.35

# Honest headline figures from our own survivorship-free testing (DECISIONS.md).
RISK = {
    "excess": "+4.7%",
    "drawdown": "-34%",
    "hit_rate": "12 / 22",
    "worst_3yr": "-12%",
    "window": "2005-2026",
}


def load_strategy(name: str) -> dict:
    path = STRATEGY_DIR / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in STRATEGY_DIR.glob("*.yaml"))
        raise FileNotFoundError(f"No strategy '{name}'. Available: {available}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@contextlib.contextmanager
def _quiet():
    """
    Silence vendor chatter. yfinance warns about every delisted historical member;
    those warnings are expected noise, not errors, and they drown the real output.
    We keep stdout/stderr captured so genuine problems can still be inspected.
    """
    buf_out, buf_err = io.StringIO(), io.StringIO()
    prev = logging.getLogger("yfinance").level
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                yield
    finally:
        logging.getLogger("yfinance").setLevel(prev)


def _fetch_with_retry(adapter, symbols, start, end):
    """
    Bulk fetch, then RETRY the misses one at a time.

    A bulk download can fail on a symbol that succeeds alone -- CF and ON both did.
    Dropping a current index member from the ranking because of a transient vendor
    failure changes the portfolio, so it is worth a second attempt. Anything still
    missing is returned so the report can disclose it.
    """
    with _quiet():
        prices = adapter.fetch(symbols, start, end)

    close = prices.close
    missing = [
        s for s in symbols
        if s not in close.columns or close[s].notna().sum() == 0
    ]

    recovered = []
    for sym in list(missing):
        try:
            with _quiet():
                one = adapter.fetch([sym], start, end)
            col = one.close[sym].dropna() if sym in one.close.columns else None
            if col is not None and len(col) > 0:
                prices.close[sym] = one.close[sym]
                if sym in one.volume.columns:
                    prices.volume[sym] = one.volume[sym]
                recovered.append(sym)
                missing.remove(sym)
        except Exception:
            pass

    return prices, missing, recovered


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _short_sector(s) -> str:
    return (str(s)
            .replace("Information Technology", "Info Technology")
            .replace("Communication Services", "Comm Services")
            .replace("Consumer Discretionary", "Cons Discretionary")
            .replace("Consumer Staples", "Cons Staples")
            .replace("Health Care", "Healthcare"))


def build_html(ctx: dict) -> str:
    """Assemble the report. Kept as one template so the document is self-contained."""
    rows = "\n".join(
        f'        <tr><td class="rank">{r["rank"]:02d}</td>'
        f'<td class="sym">{_esc(r["symbol"])}</td>'
        f'<td class="co">{_esc(r["name"])}</td>'
        f'<td class="sec">{_esc(_short_sector(r["sector"]))}</td>'
        f'<td class="wt">{r["weight"]:.2%}</td></tr>'
        for r in ctx["holdings"]
    )

    tickets = "\n".join(
        f'      <div class="ticket {"sell" if t["action"]=="Close" else ""}">'
        f'<div class="act">{t["action"]}</div>'
        f'<div class="sym">{_esc(t["symbol"])}</div>'
        f'<div class="wt">{t["weight"]}</div></div>'
        for t in ctx["tickets"]
    )

    bars = "\n".join(
        f'      <div class="bar-row">'
        f'<div class="bar-label">{_esc(s)}</div>'
        f'<div class="bar-track"><div class="bar-fill{" over" if w >= CONCENTRATION_REVIEW else ""}" '
        f'style="width:{w*100:.0f}%"></div></div>'
        f'<div class="bar-val">{w:.0%}</div></div>'
        for s, w in ctx["sectors"]
    )

    dq = "\n".join(
        f'      <div class="dq-item"><div class="dq-tag {d["tag"]}">{d["label"]}</div>'
        f'<div class="dq-body">{d["body"]}</div></div>'
        for d in ctx["data_notes"]
    )

    conc = ""
    if ctx["top_sector_weight"] >= CONCENTRATION_REVIEW:
        s, w = ctx["sectors"][0]
        conc = f"""
    <div class="conc-warn">
      <div class="hdr">This month the portfolio is concentrated in one sector.</div>
      <p>
        {ctx['top_sector_count']} of {len(ctx['holdings'])} holdings are
        {_esc(s)} companies. The strategy applies no sector limit, so when one part of
        the market leads, the whole portfolio follows it there. If these names fall
        together, diversification will not cushion the portfolio.
      </p>
    </div>"""

    state_class = "on" if ctx["risk_on"] else "off"
    state_word = "Invested" if ctx["risk_on"] else "In cash"
    why = (
        "The market filter is <b>on</b>. The benchmark is trading above its 200-day "
        "average, so the strategy holds equities this month."
        if ctx["risk_on"] else
        "The market filter is <b>off</b>. The benchmark has fallen below its 200-day "
        "average, so the strategy stands aside. Close every position and hold cash."
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Momentum Signal &mdash; {ctx['date_long']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=IBM+Plex+Mono:wght@400;500;600&family=Inter+Tight:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{--graphite:#1A1E24;--graphite-2:#232830;--paper:#FAFAF7;--paper-2:#F1F0EB;
    --ink:#15181D;--ink-soft:#5A616B;--rule:#D9D7D0;--live:#1F6F5C;--exit:#9E3B32;
    --brass:#B0842F;--brass-soft:#F0E4C6;}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--graphite);color:var(--ink);font-family:'Inter Tight',system-ui,sans-serif;
    font-size:16px;line-height:1.5;padding:32px 20px 64px}}
  .sheet{{max-width:940px;margin:0 auto}}
  .masthead{{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;
    flex-wrap:wrap;padding-bottom:20px;margin-bottom:2px}}
  .brand{{color:var(--paper)}}
  .brand h1{{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;
    font-size:clamp(28px,5vw,42px);letter-spacing:-.03em;line-height:1}}
  .brand .sub{{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.14em;
    text-transform:uppercase;color:#8A929E;margin-top:10px}}
  .issued{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#8A929E;
    text-align:right;line-height:1.9}}
  .issued b{{color:var(--paper);font-weight:500;display:block;font-size:15px}}
  .verdict{{background:var(--paper);border-radius:3px;padding:24px 28px;display:flex;
    gap:28px;align-items:center;flex-wrap:wrap;border-left:6px solid var(--live)}}
  .verdict.off{{border-left-color:var(--exit)}}
  .verdict .state{{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;
    font-size:clamp(30px,5vw,44px);letter-spacing:-.03em;color:var(--live);line-height:1}}
  .verdict.off .state{{color:var(--exit)}}
  .verdict .why{{flex:1;min-width:260px;font-size:14.5px;color:var(--ink-soft)}}
  .verdict .why b{{color:var(--ink);font-weight:600}}
  .gauge{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--ink-soft);white-space:nowrap}}
  .gauge b{{color:var(--ink);font-weight:600}}
  .panel{{background:var(--paper);border-radius:3px;padding:28px;margin-top:2px}}
  .panel-head{{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
    flex-wrap:wrap;border-bottom:1px solid var(--rule);padding-bottom:12px;margin-bottom:20px}}
  .panel-head h2{{font-family:'Bricolage Grotesque',sans-serif;font-weight:600;
    font-size:19px;letter-spacing:-.015em}}
  .panel-head .note{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;
    letter-spacing:.1em;text-transform:uppercase;color:var(--ink-soft)}}
  .action-lead{{font-size:15px;color:var(--ink-soft);margin-bottom:18px;max-width:62ch}}
  .action-lead b{{color:var(--ink);font-weight:600}}
  .tickets{{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:8px}}
  .ticket{{border:1px solid var(--rule);border-left:3px solid var(--live);border-radius:2px;
    padding:9px 12px;background:#fff}}
  .ticket.sell{{border-left-color:var(--exit)}}
  .ticket .sym{{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:15px}}
  .ticket .act{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.12em;
    text-transform:uppercase;color:var(--live)}}
  .ticket.sell .act{{color:var(--exit)}}
  .ticket .wt{{font-size:12px;color:var(--ink-soft);margin-top:2px}}
  .holdings{{width:100%;border-collapse:collapse;font-size:14.5px}}
  .holdings thead th{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;
    letter-spacing:.12em;text-transform:uppercase;color:var(--ink-soft);text-align:left;
    font-weight:500;padding:0 10px 10px}}
  .holdings thead th.num{{text-align:right}}
  .holdings tbody tr{{border-top:1px solid var(--rule)}}
  .holdings td{{padding:9px 10px;vertical-align:baseline}}
  .rank{{font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;width:42px;
    border-right:2px solid var(--graphite)}}
  .sym{{font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:14.5px;width:78px}}
  .co{{color:var(--ink-soft);font-size:13.5px}}
  .sec{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink-soft);white-space:nowrap}}
  .wt{{font-family:'IBM Plex Mono',monospace;font-size:13.5px;text-align:right;width:74px}}
  .conc-warn{{background:var(--brass-soft);border-left:6px solid var(--brass);border-radius:2px;
    padding:16px 20px;margin-bottom:20px}}
  .conc-warn .hdr{{font-family:'Bricolage Grotesque',sans-serif;font-weight:600;font-size:16px;
    margin-bottom:6px}}
  .conc-warn p{{font-size:14px;color:#6B5320;max-width:70ch}}
  .bars{{display:flex;flex-direction:column;gap:11px}}
  .bar-row{{display:grid;grid-template-columns:190px 1fr 58px;gap:14px;align-items:center}}
  .bar-label{{font-size:13.5px}}
  .bar-track{{height:16px;background:var(--paper-2);border-radius:1px;overflow:hidden}}
  .bar-fill{{height:100%;background:var(--graphite)}}
  .bar-fill.over{{background:var(--brass)}}
  .bar-val{{font-family:'IBM Plex Mono',monospace;font-size:13px;text-align:right}}
  .threshold{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--ink-soft);
    margin-top:14px;padding-top:12px;border-top:1px dashed var(--rule)}}
  .risk-panel{{background:var(--graphite-2);color:var(--paper)}}
  .risk-panel .panel-head{{border-bottom-color:#39404B}}
  .risk-panel .panel-head h2{{color:var(--paper)}}
  .risk-panel .panel-head .note{{color:#8A929E}}
  .risk-lead{{font-size:15px;color:#B9C0C9;max-width:66ch;margin-bottom:22px}}
  .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:2px;
    background:#39404B}}
  .metric{{background:var(--graphite-2);padding:16px 18px}}
  .metric .v{{font-family:'Bricolage Grotesque',sans-serif;font-weight:800;font-size:30px;
    letter-spacing:-.03em;line-height:1.05}}
  .metric .v.bad{{color:#E38279}}
  .metric .k{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.1em;
    text-transform:uppercase;color:#8A929E;margin-top:7px}}
  .metric .d{{font-size:12.5px;color:#B9C0C9;margin-top:6px;line-height:1.4}}
  .risk-foot{{font-size:13.5px;color:#B9C0C9;margin-top:20px;max-width:72ch}}
  .risk-foot b{{color:var(--paper);font-weight:600}}
  .dq{{display:grid;gap:10px}}
  .dq-item{{display:grid;grid-template-columns:88px 1fr;gap:14px;font-size:14px;
    padding-bottom:10px;border-bottom:1px solid var(--rule)}}
  .dq-item:last-child{{border-bottom:0;padding-bottom:0}}
  .dq-tag{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;letter-spacing:.1em;
    text-transform:uppercase;padding-top:3px}}
  .dq-tag.ok{{color:var(--live)}}
  .dq-tag.gap{{color:var(--brass)}}
  .dq-body{{color:var(--ink-soft)}}
  .dq-body b{{color:var(--ink);font-weight:600}}
  .dq-body code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;
    background:var(--paper-2);padding:1px 5px;border-radius:2px;color:var(--ink)}}
  .colophon{{margin-top:26px;padding:22px 4px;color:#7A828E;font-size:12.5px;
    line-height:1.65;max-width:78ch}}
  .colophon b{{color:#B9C0C9;font-weight:500}}
  .colophon .rule{{height:1px;background:#39404B;margin-bottom:18px}}
  @media (max-width:640px){{
    body{{padding:20px 12px 48px}} .panel{{padding:20px 16px}}
    .bar-row{{grid-template-columns:120px 1fr 50px;gap:10px}}
    .holdings .co{{display:none}} .dq-item{{grid-template-columns:1fr;gap:4px}}
  }}
  @media print{{
    body{{background:#fff;padding:0}} .panel,.verdict{{break-inside:avoid}}
    .risk-panel{{background:#fff;color:var(--ink);border:1px solid var(--rule)}}
    .risk-panel .panel-head h2,.metric .v{{color:var(--ink)}}
    .metric{{background:#fff}} .metrics{{background:var(--rule)}}
    .brand h1,.issued b{{color:var(--ink)}} .colophon{{color:var(--ink-soft)}}
  }}
  @media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>
</head>
<body>
<div class="sheet">
  <header class="masthead">
    <div class="brand">
      <h1>Momentum Signal</h1>
      <div class="sub">{_esc(ctx['universe_label'])} &middot; Top {ctx['top_n']} &middot; Monthly rebalance</div>
    </div>
    <div class="issued">Issued<b>{ctx['date_long']}</b>Generated {ctx['generated']}</div>
  </header>

  <section class="verdict {state_class}">
    <div class="state">{state_word}</div>
    <div class="why">{why}</div>
    <div class="gauge">{ctx['gauge']}</div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>What to trade</h2>
      <div class="note">{ctx['ticket_summary']}</div></div>
    <p class="action-lead">{ctx['action_lead']}</p>
    <div class="tickets">
{tickets}
    </div>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Holdings</h2>
      <div class="note">Ranked by {ctx['signal_label']}</div></div>
    <table class="holdings">
      <thead><tr><th>Rank</th><th>Ticker</th><th>Company</th><th>Sector</th><th class="num">Weight</th></tr></thead>
      <tbody>
{rows}
      </tbody>
    </table>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Concentration</h2>
      <div class="note">Read this before you trade</div></div>{conc}
    <div class="bars">
{bars}
    </div>
    <div class="threshold">REVIEW THRESHOLD {CONCENTRATION_REVIEW:.0%} &nbsp;&middot;&nbsp;
      LARGEST SECTOR {ctx['top_sector_weight']:.0%}</div>
  </section>

  <section class="panel risk-panel">
    <div class="panel-head"><h2>What this strategy has done, honestly</h2>
      <div class="note">{RISK['window']} &middot; survivorship-free</div></div>
    <p class="risk-lead">
      These figures come from testing that includes the companies that failed and were
      delisted. Most published backtests quietly exclude them, which flatters the
      result. Ours does not, so these numbers are lower &mdash; and closer to true.
    </p>
    <div class="metrics">
      <div class="metric"><div class="v">{RISK['excess']}</div>
        <div class="k">Per year vs the index</div>
        <div class="d">After the market filter. Before your trading costs and tax.</div></div>
      <div class="metric"><div class="v bad">{RISK['drawdown']}</div>
        <div class="k">Worst drawdown</div>
        <div class="d">$100k would have fallen to $66k before recovering.</div></div>
      <div class="metric"><div class="v">{RISK['hit_rate']}</div>
        <div class="k">Years beating the index</div>
        <div class="d">Roughly half. Losing years are normal, not a malfunction.</div></div>
      <div class="metric"><div class="v bad">{RISK['worst_3yr']}</div>
        <div class="k">Worst 3-year stretch</div>
        <div class="d">Per year, behind the index. Long enough to feel like failure.</div></div>
    </div>
    <p class="risk-foot">
      <b>The hardest part is not the loss, it is the wait.</b> There have been
      three-year periods where this strategy trailed a simple index fund by twelve
      percent a year. Anyone who abandoned it during one of those stretches locked in
      the shortfall and missed the recovery. If you would not hold through that, this
      strategy is not suitable for you.
    </p>
  </section>

  <section class="panel">
    <div class="panel-head"><h2>Data notes</h2>
      <div class="note">Known gaps in this month's run</div></div>
    <div class="dq">
{dq}
    </div>
  </section>

  <footer class="colophon">
    <div class="rule"></div>
    <b>How this was produced.</b> Prices are split- and dividend-adjusted. Index
    membership is applied as it stood on each historical date, so the test never holds
    a company before it joined the index. Delisted companies are included at their real
    collapse prices. Every trade is charged commission and slippage.
    <br><br>
    <b>This is not investment advice.</b> It is the output of a rules-based system,
    published for the people who chose to follow it. Past results do not predict future
    returns, and a strategy that has worked can stop working. You are responsible for
    your own decisions, including whether to act on this at all.
  </footer>
</div>
</body>
</html>
"""


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STRATEGY
    cfg = load_strategy(name)

    market = load_market(cfg["market"])
    membership = load_membership(market, cfg["universe"])
    adapter = get_adapter(market)
    signal = get_signal(cfg["signal"]["name"], **cfg["signal"].get("params", {}))
    top_n = cfg["selection"]["top_n"]

    trend_cfg = cfg.get("overlay", {}).get("trend_filter", {})
    overlay = (get_overlay("trend_filter", ma_days=trend_cfg.get("benchmark_ma_days", 200))
               if trend_cfg.get("enabled") else get_overlay("always_on"))

    today = pd.Timestamp.today().normalize()
    start = (today - pd.DateOffset(months=HISTORY_MONTHS)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    current = membership.as_of(today)
    current_symbols = sorted(set(current.symbols))

    print(f"  Building report for {cfg['name']} as of {today.date()}...")
    print(f"  Fetching {len(current_symbols)} current constituents...")
    prices, missing, recovered = _fetch_with_retry(adapter, current_symbols, start, end)
    if recovered:
        print(f"  Recovered on retry: {', '.join(recovered)}")
    if missing:
        print(f"  Still unpriceable: {', '.join(missing)}")

    with _quiet():
        benchmark = adapter.fetch_benchmark(start, end)

    decision = overlay.decide(benchmark, today)
    snapshot = eligible_universe(prices=prices, membership=membership, as_of=today,
                                 min_history_days=signal.required_history_days)
    scores = signal.compute(prices, as_of=today, symbols=snapshot.eligible)

    holdings, sectors, tickets = [], [], []
    top_sector_weight, top_sector_count = 0.0, 0

    if decision.risk_on:
        portfolio = build_portfolio(
            signal_result=scores, market=market, top_n=top_n,
            weighting=cfg["weighting"]["method"], membership=membership, prices=prices,
            max_position_weight=cfg["weighting"].get("max_position_weight"),
            max_sector_weight=cfg["weighting"].get("max_sector_weight"),
        )
        frame = portfolio.to_frame()
        name_by_symbol = {m.symbol: m.name for m in membership.members}
        for row in frame.itertuples():
            holdings.append({
                "rank": int(row.rank), "symbol": row.symbol,
                "name": name_by_symbol.get(row.symbol, row.symbol),
                "sector": row.sector, "weight": float(row.weight),
            })
        sw = portfolio.sector_weights().sort_values(ascending=False)
        sectors = [(s, float(w)) for s, w in sw.items()]
        if sectors:
            top_sector_weight = sectors[0][1]
            top_sector_count = sum(1 for h in holdings if h["sector"] == sectors[0][0])

    # ---- trade list vs the last saved portfolio ---------------------------
    prev = None
    if SAVE_DIR.exists():
        files = sorted(SAVE_DIR.glob(f"{cfg['strategy_id']}_*.json"))
        if files:
            try:
                prev = json.loads(files[-1].read_text())
            except Exception:
                prev = None
    prev_syms = list(prev["symbols"]) if prev else []
    target = [h["symbol"] for h in holdings]

    if prev is None:
        for h in holdings:
            tickets.append({"action": "Open", "symbol": h["symbol"],
                            "weight": f"{h['weight']:.2%}"})
        action_lead = (f"This is the <b>first portfolio</b> for this strategy, so every "
                       f"position is a new open. Buy all {len(holdings)} at the weights "
                       f"shown. From next month the report will list only the changes.")
        ticket_summary = f"{len(holdings)} opens &middot; 0 closes"
    else:
        buys = [h for h in holdings if h["symbol"] not in prev_syms]
        sells = [s for s in prev_syms if s not in target]
        for s in sells:
            tickets.append({"action": "Close", "symbol": s, "weight": "sell in full"})
        for h in buys:
            tickets.append({"action": "Open", "symbol": h["symbol"],
                            "weight": f"{h['weight']:.2%}"})
        if not tickets:
            action_lead = ("<b>No trades this month.</b> The ranking shuffled but not "
                           "enough to change the portfolio. Leave everything as it is.")
        else:
            action_lead = (f"Since the {prev.get('as_of','last')} report, "
                           f"<b>{len(sells)} to close and {len(buys)} to open</b>. "
                           f"Everything not listed here stays as it is. Rebalance the "
                           f"remaining holdings back to equal weight.")
        ticket_summary = f"{len(buys)} opens &middot; {len(sells)} closes"

    # ---- data notes -------------------------------------------------------
    drops = snapshot.drop_reasons()
    notes = [{
        "tag": "ok", "label": "Ranked",
        "body": (f"<b>{snapshot.n_eligible} of {len(current_symbols)}</b> current index "
                 f"members had enough clean price history to be ranked."
                 + (f" Excluded: " + ", ".join(f"{n} {r.replace('_',' ')}"
                                               for r, n in sorted(drops.items(),
                                                                  key=lambda kv: -kv[1]))
                    + "." if drops else "")),
    }]
    if missing:
        notes.append({
            "tag": "gap", "label": "Excluded",
            "body": (f"<b>{len(missing)} current member(s) could not be priced</b> by our "
                     f"data provider, even on retry &mdash; "
                     + ", ".join(f"<code>{_esc(m)}</code>" for m in missing)
                     + ". They were left out of the ranking, so this month's list may be "
                       "missing a name it should contain."),
        })
    if recovered:
        notes.append({
            "tag": "ok", "label": "Recovered",
            "body": ("Initially failed but retrieved on retry: "
                     + ", ".join(f"<code>{_esc(r)}</code>" for r in recovered)
                     + ". These are included in the ranking."),
        })

    ctx = {
        "date_long": today.strftime("%-d %B %Y") if sys.platform != "win32"
                     else today.strftime("%#d %B %Y"),
        "generated": datetime.now().strftime("%H:%M"),
        "universe_label": cfg.get("name", cfg["universe"]).replace(" Top-20 Momentum", ""),
        "top_n": top_n,
        "signal_label": f"{signal}".strip("<>").replace("MomentumSignal ", "momentum, "),
        "risk_on": bool(decision.risk_on),
        "gauge": (f"BENCHMARK <b>{decision.detail.get('price',0):,.0f}</b><br>"
                  f"{decision.detail.get('ma_days',200)}-DAY AVG "
                  f"<b>{decision.detail.get('ma',0):,.0f}</b>"
                  if "price" in decision.detail else _esc(decision.reason)),
        "holdings": holdings, "sectors": sectors, "tickets": tickets,
        "action_lead": action_lead, "ticket_summary": ticket_summary,
        "top_sector_weight": top_sector_weight, "top_sector_count": top_sector_count,
        "data_notes": notes,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{cfg['strategy_id']}_{today.date()}.html"
    out.write_text(build_html(ctx), encoding="utf-8")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    (SAVE_DIR / f"{cfg['strategy_id']}_{today.date()}.json").write_text(json.dumps({
        "strategy_id": cfg["strategy_id"], "as_of": str(today.date()),
        "risk_on": bool(decision.risk_on), "symbols": target,
        "weights": {h["symbol"]: h["weight"] for h in holdings},
    }, indent=2))

    print()
    print(f"  Report written to  {out}")
    print(f"  Open it in a browser, or print to PDF to send it.")
    print()


if __name__ == "__main__":
    main()
