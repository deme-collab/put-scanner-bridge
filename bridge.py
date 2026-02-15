"""
Put Scanner Bridge — Local API Server
Connects the GitHub Pages dashboard to live market data.

Combines:
  - yfinance: stock price, options chain, IV
  - K-means support detection (from discord_stock_scanner)
  - Black-Scholes greeks engine

Run:
  pip install fastapi uvicorn yfinance scikit-learn numpy pandas
  python bridge.py

Dashboard calls:
  http://localhost:8000/scan/AAPL
  http://localhost:8000/chain/AAPL
  http://localhost:8000/health
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional

try:
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except ImportError:
    KMEANS_AVAILABLE = False
    print("⚠️  scikit-learn not installed. K-means support detection disabled.")
    print("   pip install scikit-learn")

app = FastAPI(title="Put Scanner Bridge", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── BLACK-SCHOLES ENGINE ──────────────────────────────────

def norm_cdf(x):
    a1, a2, a3, a4, a5, p = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429, 0.3275911
    sign = -1 if x < 0 else 1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t*math.exp(-x*x)
    return 0.5 * (1 + sign * y)

def norm_pdf(x):
    return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)

def bs_put_price(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return max(K - S, 0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*iv*iv)*T) / (iv*sqrtT)
    d2 = d1 - iv*sqrtT
    return K * math.exp(-r*T) * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return -1.0 if S < K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*iv*iv)*T) / (iv*sqrtT)
    return norm_cdf(d1) - 1

def bs_gamma(S, K, T, r, iv):
    if T <= 0 or iv <= 0 or S <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*iv*iv)*T) / (iv*sqrtT)
    return norm_pdf(d1) / (S * iv * sqrtT)

def bs_theta(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*iv*iv)*T) / (iv*sqrtT)
    d2 = d1 - iv*sqrtT
    term1 = -(S * norm_pdf(d1) * iv) / (2 * sqrtT)
    return (term1 + r * K * math.exp(-r*T) * norm_cdf(-d2)) / 365

def bs_vega(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S/K) + (r + 0.5*iv*iv)*T) / (iv*sqrtT)
    return S * norm_pdf(d1) * sqrtT / 100

def compute_greeks(S, K, T, r, iv):
    return {
        "price": round(bs_put_price(S, K, T, r, iv), 4),
        "delta": round(bs_delta(S, K, T, r, iv), 4),
        "gamma": round(bs_gamma(S, K, T, r, iv), 6),
        "theta": round(bs_theta(S, K, T, r, iv), 4),
        "vega": round(bs_vega(S, K, T, r, iv), 4),
    }


# ─── K-MEANS SUPPORT DETECTION ────────────────────────────

def find_swing_lows(data: pd.DataFrame, window: int = 5) -> list:
    """Find swing low points in price data."""
    lows = data['Low'].values
    swing_lows = []
    for i in range(window, len(lows) - window):
        if all(lows[i] <= lows[i-window:i]) and all(lows[i] <= lows[i+1:i+window+1]):
            swing_lows.append(lows[i])
    return swing_lows

def find_swing_highs(data: pd.DataFrame, window: int = 5) -> list:
    """Find swing high points (resistance candidates)."""
    highs = data['High'].values
    swing_highs = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[i-window:i]) and all(highs[i] >= highs[i+1:i+window+1]):
            swing_highs.append(highs[i])
    return swing_highs

def kmeans_cluster(prices: np.array, max_k: int = 5) -> list:
    """Cluster prices using K-means with elbow detection."""
    if not KMEANS_AVAILABLE or len(prices) < 2:
        return list(prices)
    
    prices = np.array(prices).reshape(-1, 1)
    
    # Elbow method for optimal K
    K_range = range(1, min(max_k + 1, len(prices)))
    wcss = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(prices)
        wcss.append(km.inertia_)
    
    # Simple elbow detection
    optimal_k = 1
    if len(wcss) > 1:
        for i in range(len(wcss) - 1):
            if wcss[i] - wcss[i+1] < 0.1 * wcss[0]:
                optimal_k = i + 1
                break
        else:
            optimal_k = len(wcss)
    
    optimal_k = max(1, min(optimal_k, len(prices)))
    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    km.fit(prices)
    return sorted(km.cluster_centers_.flatten().tolist())

def get_support_resistance(symbol: str) -> Dict:
    """
    Detect support and resistance levels using K-means clustering
    on swing lows/highs from 60 days of hourly data.
    """
    try:
        tk = yf.Ticker(symbol)
        
        # Hourly data for intraday support
        hourly = tk.history(period="60d", interval="1h")
        # Daily data for stronger levels
        daily = tk.history(period="6mo", interval="1d")
        
        supports = []
        resistances = []
        
        # --- Intraday supports (hourly swing lows) ---
        if len(hourly) > 20:
            swing_lows = find_swing_lows(hourly, window=5)
            if len(swing_lows) >= 2:
                clusters = kmeans_cluster(swing_lows, max_k=4)
                for center in clusters:
                    touches = sum(1 for p in swing_lows if abs(p - center) / center < 0.015)
                    supports.append({
                        "price": round(center, 2),
                        "touches": touches,
                        "strength": min(touches * 2, 10),
                        "timeframe": "intraday",
                        "type": "major" if touches >= 4 else "minor"
                    })
        
        # --- Daily supports (stronger signal) ---
        if len(daily) > 20:
            daily_lows = find_swing_lows(daily, window=3)
            if len(daily_lows) >= 2:
                clusters = kmeans_cluster(daily_lows, max_k=4)
                for center in clusters:
                    touches = sum(1 for p in daily_lows if abs(p - center) / center < 0.02)
                    supports.append({
                        "price": round(center, 2),
                        "touches": touches,
                        "strength": min(touches * 3, 10),
                        "timeframe": "daily",
                        "type": "major" if touches >= 3 else "minor"
                    })
        
        # --- Resistance levels ---
        if len(hourly) > 20:
            swing_highs = find_swing_highs(hourly, window=5)
            if len(swing_highs) >= 2:
                clusters = kmeans_cluster(swing_highs, max_k=4)
                for center in clusters:
                    touches = sum(1 for p in swing_highs if abs(p - center) / center < 0.015)
                    resistances.append({
                        "price": round(center, 2),
                        "touches": touches,
                        "strength": min(touches * 2, 10),
                        "timeframe": "intraday"
                    })
        
        # Sort by price
        supports.sort(key=lambda x: x["price"], reverse=True)
        resistances.sort(key=lambda x: x["price"])
        
        return {"supports": supports[:6], "resistances": resistances[:4]}
    
    except Exception as e:
        print(f"Support/resistance error for {symbol}: {e}")
        return {"supports": [], "resistances": []}


# ─── SCORING ENGINE ────────────────────────────────────────

def score_contract(S, K, dte, iv, r, vix, daily_change, supports=None):
    """
    Score a put contract. Same logic as dashboard + support confluence bonus.
    """
    T = dte / 365
    g = compute_greeks(S, K, T, r, iv)
    
    if vix >= 35:
        return {"score": 0, "greeks": g, "ann_return": 0, "breakdown": {}, "support_boost": 0}
    
    sc = 0
    
    # Regime (20)
    regime = 20 if vix < 15 else 17 if vix < 20 else 12 if vix < 25 else 6 if vix < 28 else 2
    sc += regime
    
    # Delta precision (18)
    dd = abs(abs(g["delta"]) - 0.20)
    delta_pts = 18 if dd < 0.02 else 14 if dd < 0.04 else 10 if dd < 0.06 else 5 if dd < 0.10 else 1
    sc += delta_pts
    
    # Premium (14)
    ann = (g["price"] / K) * (365 / max(dte, 1))
    prem_pts = 14 if 0.08 <= ann <= 0.15 else 10 if 0.06 <= ann <= 0.25 else 3 if ann < 0.06 else 6 if ann <= 0.35 else 0
    sc += prem_pts
    
    # Distance (14)
    dist = (S - K) / S
    dist_pts = 14 if 0.05 <= dist <= 0.10 else 10 if 0.10 < dist <= 0.15 else 6 if 0.03 <= dist < 0.05 else 0 if dist < 0.03 else 5
    sc += dist_pts
    
    # Theta (12)
    td = abs(g["theta"])
    theta_pts = 12 if td > 0.05 else 9 if td > 0.03 else 5 if td > 0.01 else 2
    sc += theta_pts
    
    # Dip bonus (16)
    dip_pts = 16 if daily_change < -2 else 13 if daily_change < -1 else 10 if daily_change < -0.5 else 5 if daily_change < 0 else 2 if daily_change < 0.5 else -2 if daily_change < 1.5 else -8
    sc += dip_pts
    
    # Gamma penalty
    gamma_pen = -6 if dte < 14 else -3 if dte < 21 else 0
    sc += gamma_pen
    
    # K-means support confluence bonus (up to +15)
    support_boost = 0
    if supports:
        for sup in supports:
            pct_diff = abs(K - sup["price"]) / K
            if pct_diff < 0.02:  # Strike within 2% of support
                boost = min(15, sup["strength"] * 3)
                if boost > support_boost:
                    support_boost = boost
    sc += support_boost
    
    return {
        "score": max(0, min(100, round(sc))),
        "greeks": g,
        "ann_return": round(ann, 4),
        "breakdown": {
            "regime": regime,
            "delta": delta_pts,
            "premium": prem_pts,
            "distance": dist_pts,
            "theta": theta_pts,
            "dip": max(dip_pts, 0),
            "gamma": gamma_pen,
            "support": support_boost,
        },
        "support_boost": support_boost,
    }


# ─── VIX FETCH ─────────────────────────────────────────────

def get_vix():
    try:
        vix = yf.Ticker("^VIX")
        info = vix.info
        return info.get("regularMarketPrice", info.get("previousClose", 18))
    except:
        return 18  # fallback


# ─── API ENDPOINTS ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "kmeans": KMEANS_AVAILABLE, "time": datetime.now().isoformat()}


@app.get("/scan/{ticker}")
def scan_ticker(ticker: str, dte: int = 35):
    """
    Full scan: price + options chain + greeks + K-means support + scoring.
    This is the main endpoint the dashboard calls.
    """
    ticker = ticker.upper()
    r = 0.045  # risk-free rate
    
    try:
        tk = yf.Ticker(ticker)
        
        # 1. Current price + daily change
        hist = tk.history(period="5d", interval="1d")
        if len(hist) < 2:
            return {"error": f"No price data for {ticker}"}
        
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2])
        daily_change = ((current_price - prev_close) / prev_close) * 100
        
        # 2. VIX
        vix = get_vix()
        
        # 3. K-means support/resistance
        levels = get_support_resistance(ticker)
        
        # 4. Options chain — find expiration closest to target DTE
        exps = tk.options
        if not exps:
            return {"error": f"No options for {ticker}"}
        
        target_date = datetime.now() + timedelta(days=dte)
        best_exp = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - target_date).days))
        actual_dte = (datetime.strptime(best_exp, "%Y-%m-%d") - datetime.now()).days
        
        chain = tk.option_chain(best_exp)
        puts = chain.puts
        
        # Filter: 2% to 25% OTM
        puts = puts[
            (puts['strike'] < current_price * 0.98) & 
            (puts['strike'] > current_price * 0.75)
        ].copy()
        
        # 5. Score each contract
        contracts = []
        for _, row in puts.iterrows():
            K = float(row['strike'])
            iv = float(row.get('impliedVolatility', 0))
            if iv <= 0:
                continue
            
            T = actual_dte / 365
            g = compute_greeks(current_price, K, T, r, iv)
            
            # Skip if delta is outside useful range
            if abs(g["delta"]) < 0.03 or abs(g["delta"]) > 0.50:
                continue
            
            res = score_contract(current_price, K, actual_dte, iv, r, vix, daily_change, levels["supports"])
            
            # Check if strike sits on support
            on_support = False
            support_name = ""
            for sup in levels["supports"]:
                if abs(K - sup["price"]) / K < 0.02:
                    on_support = True
                    support_name = f"{sup['timeframe']} ({sup['touches']} touches)"
                    break
            
            contracts.append({
                "strike": K,
                "bid": float(row.get('bid', 0)),
                "ask": float(row.get('ask', 0)),
                "last": float(row.get('lastPrice', 0)),
                "iv": round(iv * 100, 1),
                "volume": int(row.get('volume', 0) or 0),
                "oi": int(row.get('openInterest', 0) or 0),
                "bs_price": g["price"],
                "delta": g["delta"],
                "gamma": g["gamma"],
                "theta": g["theta"],
                "vega": g["vega"],
                "score": res["score"],
                "ann_return": round(res["ann_return"] * 100, 1),
                "distance": round((current_price - K) / current_price * 100, 1),
                "breakdown": res["breakdown"],
                "on_support": on_support,
                "support_info": support_name,
            })
        
        # Sort by score
        contracts.sort(key=lambda x: x["score"], reverse=True)
        
        # Find the recommended contract (highest score)
        rec = contracts[0] if contracts else None
        
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "daily_change": round(daily_change, 2),
            "vix": round(vix, 1),
            "expiration": best_exp,
            "dte": actual_dte,
            "supports": levels["supports"],
            "resistances": levels["resistances"],
            "contracts": contracts[:20],  # Top 20
            "recommended": rec,
            "warnings": _get_warnings(current_price, vix, iv=rec["iv"] if rec else 0, dte=actual_dte),
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/chain/{ticker}")
def get_chain(ticker: str, exp: str = None):
    """
    Raw options chain for a ticker. Lighter weight than /scan.
    """
    ticker = ticker.upper()
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        
        exps = tk.options
        if not exps:
            return {"error": "No options found"}
        
        if exp and exp in exps:
            target_exp = exp
        else:
            target_exp = exps[min(3, len(exps) - 1)]
        
        chain = tk.option_chain(target_exp)
        puts = chain.puts
        puts = puts[(puts['strike'] > current_price * 0.70) & (puts['strike'] < current_price * 1.02)]
        
        data = []
        for _, row in puts.iterrows():
            data.append({
                "strike": float(row['strike']),
                "bid": float(row.get('bid', 0)),
                "ask": float(row.get('ask', 0)),
                "last": float(row.get('lastPrice', 0)),
                "iv": round(float(row.get('impliedVolatility', 0)) * 100, 1),
                "volume": int(row.get('volume', 0) or 0),
                "oi": int(row.get('openInterest', 0) or 0),
            })
        
        return {
            "ticker": ticker,
            "price": round(current_price, 2),
            "expiration": target_exp,
            "expirations": list(exps),
            "chain": data,
        }
    
    except Exception as e:
        return {"error": str(e)}


@app.get("/supports/{ticker}")
def get_supports(ticker: str):
    """Just the K-means support/resistance levels."""
    ticker = ticker.upper()
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d", interval="1d")
        price = float(hist['Close'].iloc[-1]) if len(hist) > 0 else 0
        levels = get_support_resistance(ticker)
        return {"ticker": ticker, "price": round(price, 2), **levels}
    except Exception as e:
        return {"error": str(e)}


def _get_warnings(price, vix, iv, dte):
    """Generate warning flags for the dashboard."""
    warnings = []
    if vix >= 28:
        warnings.append({"type": "regime", "msg": f"VIX at {vix} — elevated regime risk"})
    if iv > 50:
        warnings.append({"type": "iv", "msg": f"IV {iv}% is very high — check for earnings or catalysts"})
    if dte < 21:
        warnings.append({"type": "dte", "msg": f"Only {dte} DTE — gamma risk elevated"})
    return warnings


# ─── RUN ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"\n  $ Put Scanner Bridge v1.0 — port {port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
