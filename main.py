# main.py
"""
Streamlit app: Options Greeks with Kite Connect
- Guided Kite login flow (login URL -> paste request_token -> generate access_token -> hides login UI)
- Build option chain from kite.instruments()
- ATM ± N strike dropdown
- Auto-fill IV by inverting BS from LTP where possible
- Refresh instruments & update IVs
- Portfolio CRUD + CSV export
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq
import json
import time

# Kite import
try:
    from kiteconnect import KiteConnect
    HAS_KITE = True
except Exception:
    HAS_KITE = False


def check_password():
    def password_entered():
        if st.session_state["password"] == "Rudraj@911":   # <-- change if needed
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Enter password", type="password", on_change=password_entered, key="password")
        st.error("Password incorrect ❌")
        return False
    else:
        return True

if not check_password():
    st.stop()   




st.set_page_config(page_title="Options Greeks - Kite Connect", layout="wide")
st.title("Options Portfolio Greeks — Kite Connect")

# ---------------------
# Black-Scholes helpers
# ---------------------
def bs_price(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def implied_vol_from_price(market_price, S, K, T, r, option_type, sigma_lo=1e-4, sigma_hi=5.0):
    try:
        def f(sig):
            return bs_price(S, K, T, r, sig, option_type) - market_price
        f_lo = f(sigma_lo)
        f_hi = f(sigma_hi)
        if f_lo * f_hi > 0:
            sigma_hi2 = 10.0
            f_hi2 = f(sigma_hi2)
            if f_lo * f_hi2 > 0:
                return None
            sigma_hi = sigma_hi2
        vol = brentq(f, sigma_lo, sigma_hi, maxiter=100, xtol=1e-6)
        return float(vol)
    except Exception:
        return None

def bs_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = norm.cdf(d1) if option_type == "CE" else norm.cdf(d1) - 1.0
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    if option_type == "CE":
        theta = - (S * norm.pdf(d1) * sigma) / (2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2)
    else:
        theta = - (S * norm.pdf(d1) * sigma) / (2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return float(delta), float(theta), float(gamma), float(vega)

# ---------------------
# Sidebar: Kite login & credentials (user-friendly hide-after-login)
# ---------------------
st.sidebar.header("Kite Connect — Login / Access Token")

# You can choose to hardcode API_KEY/API_SECRET here or leave blank and use the login URL flow.
API_KEY = "afzia78cwraaod5x"
API_SECRET = "b527807j5ilcndjp5u2jhu9znrjxz35e"

# Paste an existing access token or JSON (optional)
pasted_token_area = st.sidebar.text_area("Paste existing access_token (string or JSON with key 'access_token')", value="", height=120)

# session initialization
if 'kite' not in st.session_state:
    st.session_state.kite = None
if 'access_token' not in st.session_state:
    st.session_state.access_token = None
if 'instruments' not in st.session_state:
    st.session_state.instruments = None
if 'instruments_fetched_at' not in st.session_state:
    st.session_state.instruments_fetched_at = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# If user pasted token JSON/string, use it and hide login UI
if pasted_token_area.strip():
    try:
        obj = json.loads(pasted_token_area.strip())
        if isinstance(obj, dict) and obj.get("access_token"):
            st.session_state.access_token = obj.get("access_token")
        elif isinstance(obj, str):
            st.session_state.access_token = obj
    except Exception:
        # not JSON, treat as raw token
        st.session_state.access_token = pasted_token_area.strip()

# Helper to init Kite client from access token
def init_kite_client(api_key, access_token):
    try:
        kite_client = KiteConnect(api_key=api_key)
        kite_client.set_access_token(access_token)
        return kite_client
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Kite client: {e}")
        return None

# If we have an access token already in session_state, show success and hide the login area
if 'access_token' in st.session_state and st.session_state.access_token:
    # initialize kite client if not already initialized and API_KEY provided
    if st.session_state.kite is None and API_KEY:
        try:
            st.session_state.kite = init_kite_client(API_KEY, st.session_state.access_token)
        except Exception:
            st.session_state.kite = None

    st.sidebar.success("Access token available — login section hidden.")
    st.sidebar.write("If you want to generate a new token, clear session or restart the app.")
    if st.sidebar.button("Clear saved access token"):
        st.session_state.access_token = None
        st.session_state.kite = None
        st.experimental_rerun()
else:
    # No access token — show guided login flow
    st.sidebar.markdown("### Generate Access Token (guided)")
    if API_KEY:
        try:
            kite_for_login = KiteConnect(api_key=API_KEY)
            login_url = kite_for_login.login_url()
            st.sidebar.write("1. Open this login URL and login to Kite:")
            st.sidebar.markdown(f"[Open Kite login URL]({login_url})")
            st.sidebar.write("2. After login, you'll be redirected to your redirect URI with `request_token` in the URL.")
            request_token_input = st.sidebar.text_input("Paste `request_token` (from redirect URL)", value="")
            if st.sidebar.button("Get Access Token"):
                if not request_token_input:
                    st.sidebar.error("Paste request_token first.")
                elif not API_SECRET:
                    st.sidebar.error("Enter API Secret to exchange request_token.")
                else:
                    try:
                        data = kite_for_login.generate_session(request_token_input, api_secret=API_SECRET)
                        access_token = data.get("access_token")
                        if access_token:
                            st.session_state.access_token = access_token
                            # save token to file for convenience (local)
                            try:
                                with open("access_token.txt", "w") as f:
                                    f.write(access_token)
                            except Exception:
                                pass
                            # initialize kite client
                            st.session_state.kite = init_kite_client(API_KEY, access_token)
                            st.sidebar.success("✅ Access token generated and stored in session.")
                            st.sidebar.code(f"Access token saved to session (and access_token.txt if writable).")
                            st.experimental_rerun()
                        else:
                            st.sidebar.error(f"No access_token found in response: {data}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to generate session: {e}")
        except Exception:
            st.sidebar.error("KiteConnect init failed — check API Key and internet connectivity.")
    else:
        st.sidebar.info("Enter API Key above to enable login flow, or paste an existing access token.")

# ---------------------
# Instrument & display controls
# ---------------------
st.sidebar.markdown("---")
st.sidebar.header("Instrument & strike settings")
market_choice = st.sidebar.selectbox("Market type", options=["NIFTY", "BANKNIFTY", "STOCK"], index=0)
if market_choice == "STOCK":
    symbol_input = st.sidebar.text_input("Stock symbol (e.g. RELIANCE) or instrument_key", value="RELIANCE").strip().upper()
else:
    symbol_input = st.sidebar.selectbox("Index", options=["NIFTY", "BANKNIFTY"], index=0)

strike_step = st.sidebar.number_input("Strike step", min_value=1, max_value=2000, value=50, step=1)
num_strikes_each_side = st.sidebar.number_input("ATM ± N strikes", min_value=1, max_value=50, value=6, step=1)
manual_spot = st.sidebar.number_input("Manual spot override (0 to use fetched)", value=0.0, format="%.2f")
r_rate = st.sidebar.number_input("Risk-free rate (decimal)", value=0.07, format="%.4f")

# optional: upload instruments CSV if kite.instruments() not available
st.sidebar.markdown("---")
inst_file = st.sidebar.file_uploader("Upload instruments CSV (optional)", type=['csv'])

if inst_file is not None:
    try:
        df_inst = pd.read_csv(inst_file)
        st.session_state.instruments = df_inst.to_dict('records')
        st.sidebar.success("Instruments loaded from CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to load instruments CSV: {e}")

# Attempt to auto-fetch instruments if kite client present and not yet loaded
def fetch_instruments_via_kite():
    try:
        inst_list = st.session_state.kite.instruments()
        st.session_state.instruments = inst_list
        st.session_state.instruments_fetched_at = datetime.now().isoformat()
        return inst_list
    except Exception as e:
        st.warning(f"Failed to fetch instruments from Kite: {e}")
        return None

if st.session_state.kite and (not st.session_state.instruments):
    with st.spinner("Fetching instruments from Kite (may take ~8-12s)..."):
        fetch_instruments_via_kite()

def build_option_chain(instruments, underlying_symbol, expiry_date_iso=None):
    chain = {}
    if not instruments:
        return chain
    usym = underlying_symbol.upper()
    for inst in instruments:
        try:
            trad = (inst.get('tradingsymbol') or '').upper()
            name = (inst.get('name') or '').upper()
            expiry = inst.get('expiry') or inst.get('expiry_date') or inst.get('expiryDate') or ''
            if expiry_date_iso:
                if not str(expiry).startswith(expiry_date_iso):
                    continue
            if market_choice in ("NIFTY","BANKNIFTY"):
                if usym not in trad and usym not in name:
                    continue
            else:
                if usym not in trad and usym not in name:
                    continue
            strike = inst.get('strike') or inst.get('strike_price') or inst.get('strikePrice') or 0
            try:
                strike = int(float(strike))
            except:
                continue
            tradsym = inst.get('tradingsymbol') or inst.get('symbol') or ''
            if 'CE' in tradsym:
                side = 'CE'
            elif 'PE' in tradsym:
                side = 'PE'
            else:
                continue
            chain.setdefault(strike, {})[side] = inst
        except Exception:
            continue
    return chain

def find_option_ltp(kite_client, inst):
    if not kite_client or not inst:
        return None
    exc = inst.get('exchange') or 'NFO'
    trad = inst.get('tradingsymbol') or inst.get('symbol') or None
    if not trad:
        return None
    key = f"{exc}:{trad}"
    try:
        res = kite_client.ltp(key)
        if isinstance(res, dict):
            for v in res.values():
                return v.get('last_price') or v.get('lastTradedPrice') or v.get('ltp')
    except Exception:
        return None
    return None

# ---------------------
# Add position form (correct usage)
# ---------------------
st.markdown("## Add Option Position")
with st.form(key="add_position_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        # expiry dropdown built from instruments
        expiry_days = None
        expiry_iso = None
        display_expiries = []
        expiry_map = {}
        if st.session_state.instruments:
            exps = set()
            for inst in st.session_state.instruments:
                trad = (inst.get('tradingsymbol') or '').upper()
                name = (inst.get('name') or '').upper()
                if market_choice in ("NIFTY","BANKNIFTY"):
                    if symbol_input not in trad and symbol_input not in name:
                        continue
                else:
                    if symbol_input not in trad and symbol_input not in name:
                        continue
                exp = inst.get('expiry') or inst.get('expiry_date') or inst.get('expiryDate') or ''
                if exp:
                    try:
                        iso = str(exp)[:10]
                        exps.add(iso)
                    except:
                        continue
            exps = sorted(list(exps))
            for e in exps:
                try:
                    dt = datetime.strptime(e[:10], "%Y-%m-%d").date()
                    disp = dt.strftime("%d-%b-%Y").upper()
                except:
                    disp = str(e)
                display_expiries.append(disp)
                expiry_map[disp] = e[:10]
        if display_expiries:
            expiry_display = st.selectbox("Expiry", options=display_expiries, key="expiry_select")
            expiry_iso = expiry_map.get(expiry_display)
            try:
                expiry_days = max(0, (datetime.strptime(expiry_iso, "%Y-%m-%d").date() - date.today()).days)
            except:
                expiry_days = st.number_input("Days to expiry (calc fail)", min_value=0, max_value=3650, value=7, key="expiry_days_input")
        else:
            expiry_days = st.number_input("Days to Expiry (instruments not loaded)", min_value=0, max_value=3650, value=7, key="expiry_days_input")
            expiry_iso = None

        # determine spot: try kite.ltp then manual fallback
        spot = None
        if st.session_state.kite:
            try:
                idx_key = f"NSE:{symbol_input}" if market_choice in ("NIFTY","BANKNIFTY") else f"NSE:{symbol_input}"
                l = st.session_state.kite.ltp(idx_key)
                for v in l.values():
                    spot = v.get('last_price') or v.get('lastTradedPrice') or v.get('ltp')
                    break
            except Exception:
                spot = None
        if spot is None:
            spot = st.number_input("Underlying spot (fallback)", value=25000.0, key="spot_fallback")
        else:
            st.write(f"Detected spot: {spot:.2f}")

        atm_val = int(round(spot / strike_step) * strike_step)
        strikes = [atm_val + n*strike_step for n in range(-int(num_strikes_each_side), int(num_strikes_each_side)+1)]
        strikes = sorted(list(dict.fromkeys([s for s in strikes if s > 0])))
        strike_selected = st.selectbox("Strike", options=strikes, index=len(strikes)//2, key="strike_select")
        opt_type = st.selectbox("Option Type", options=["CE","PE"], key="opt_type_select")

    with col2:
        # try auto-IV
        auto_iv = None
        auto_price = None
        if expiry_iso and st.session_state.instruments and st.session_state.kite:
            chain = build_option_chain(st.session_state.instruments, symbol_input, expiry_date_iso=expiry_iso)
            ent = chain.get(strike_selected, {}).get(opt_type)
            if ent:
                ltp_val = find_option_ltp(st.session_state.kite, ent)
                if ltp_val is not None:
                    T = max(0, expiry_days)/365.0
                    iv = implied_vol_from_price(float(ltp_val), float(spot), strike_selected, T, r=0.07, option_type=opt_type)
                    if iv is not None:
                        auto_iv = iv
                        auto_price = ltp_val

        if auto_iv is not None:
            sigma = st.number_input("Implied Volatility (decimal) — auto-filled", value=float(auto_iv), format="%.4f", min_value=0.0001, key="sigma_input")
            st.caption(f"Auto-filled IV from LTP={auto_price:.2f}")
        else:
            sigma = st.number_input("Implied Volatility (decimal)", value=0.20, format="%.4f", min_value=0.0001, key="sigma_input")
            if not st.session_state.instruments:
                st.caption("Instruments not loaded — upload CSV or use Refresh.")

        lots = st.number_input("Lots", min_value=1, max_value=10000, value=1, key="lots_input")
        lot_size = st.number_input("Lot size", min_value=1, max_value=10000, value=50, key="lot_size_input")

    with col3:
        direction = st.selectbox("Direction", options=["Long","Short"], key="direction_select")
        add_clicked = st.form_submit_button(label="Add Position")

    st.write(f"Using spot: {spot:.2f} | Expiry in {expiry_days} days")

# After form submit
if 'add_clicked' not in st.session_state:
    st.session_state.add_clicked = False
if add_clicked:
    st.session_state.add_clicked = True
    pos = {
        'expiry_days': int(expiry_days),
        'strike': int(strike_selected),
        'type': opt_type,
        'sigma': float(sigma),
        'lots': int(lots),
        'lot_size': int(lot_size),
        'direction': direction
    }
    st.session_state.portfolio.append(pos)
    st.success(f"Added {direction} {opt_type} {strike_selected} exp {expiry_days}d (IV {sigma:.4f})")

# ---------------------
# Refresh instruments & update IVs
# ---------------------
if st.button("Refresh instruments & update IVs"):
    if st.session_state.kite and HAS_KITE:
        with st.spinner("Fetching instruments from Kite..."):
            try:
                insts = st.session_state.kite.instruments()
                st.session_state.instruments = insts
                st.session_state.instruments_fetched_at = datetime.now().isoformat()
                st.success("Instruments fetched")
            except Exception as e:
                st.error(f"Failed to fetch instruments: {e}")
    else:
        st.warning("Kite client not initialized or kiteconnect not available.")

    # update IVs
    if st.session_state.portfolio and st.session_state.instruments and st.session_state.kite:
        updated = 0
        for p in st.session_state.portfolio:
            try:
                # best-effort match expiry iso
                expiry_iso = None
                for inst in st.session_state.instruments:
                    exp = inst.get('expiry') or inst.get('expiry_date') or inst.get('expiryDate') or ''
                    if exp and str(exp).startswith(str((date.today() + timedelta(days=int(p['expiry_days']))).year)):
                        expiry_iso = str(exp)[:10]
                        break
                if expiry_iso:
                    chain = build_option_chain(st.session_state.instruments, symbol_input, expiry_date_iso=expiry_iso)
                    ent = chain.get(p['strike'], {}).get(p['type'])
                    if ent:
                        ltp = find_option_ltp(st.session_state.kite, ent)
                        if ltp:
                            # get underlying spot
                            try:
                                idx_key = f"NSE:{symbol_input}" if market_choice in ("NIFTY","BANKNIFTY") else f"NSE:{symbol_input}"
                                sres = st.session_state.kite.ltp(idx_key)
                                spot_val = None
                                for v in sres.values():
                                    spot_val = v.get('last_price') or v.get('lastTradedPrice') or v.get('ltp')
                                    break
                            except Exception:
                                spot_val = None
                            if spot_val:
                                T = max(0, p['expiry_days'])/365.0
                                iv_new = implied_vol_from_price(float(ltp), float(spot_val), p['strike'], T, r=0.07, option_type=p['type'])
                                if iv_new is not None:
                                    p['sigma'] = float(iv_new)
                                    updated += 1
            except Exception:
                continue
        st.info(f"Updated IVs for {updated} positions where available.")

# ---------------------
# Portfolio display
# ---------------------
def build_display_df(port):
    rows = []
    # attempt to fetch spot once
    spot_val = None
    if st.session_state.kite:
        try:
            idx_key = f"NSE:{symbol_input}" if market_choice in ("NIFTY","BANKNIFTY") else f"NSE:{symbol_input}"
            res = st.session_state.kite.ltp(idx_key)
            for v in res.values():
                spot_val = v.get('last_price') or v.get('lastTradedPrice') or v.get('ltp')
                break
        except Exception:
            spot_val = None
    if spot_val is None:
        spot_val = 25000.0

    for p in port:
        S = spot_val
        T = max(p['expiry_days'], 0)/365.0
        delta, theta, gamma, vega = bs_greeks(S, p['strike'], T, 0.07, p['sigma'], p['type'])
        sign = 1.0 if p.get('direction','Long') == 'Long' else -1.0
        mult = p['lots'] * p['lot_size']
        rows.append({
            'expiry_days': p['expiry_days'],
            'expiry_date': (date.today() + timedelta(days=int(p['expiry_days']))).strftime("%d-%b-%Y").upper(),
            'strike': p['strike'],
            'type': p['type'],
            'direction': p['direction'],
            'sigma': p['sigma'],
            'lots': p['lots'],
            'lot_size': p['lot_size'],
            'delta_per_lot': delta,
            'theta_per_lot': theta,
            'gamma_per_lot': gamma,
            'vega_per_lot': vega,
            'total_delta': sign * delta * mult,
            'total_theta': sign * theta * mult,
            'total_gamma': sign * gamma * mult,
            'total_vega': sign * vega * mult
        })
    return pd.DataFrame(rows)

st.markdown("---")
st.header("Portfolio & Greeks")

if st.session_state.portfolio:
    df = build_display_df(st.session_state.portfolio)
    st.subheader("Positions")
    st.dataframe(df.style.format({
        'sigma':'{:.4f}',
        'delta_per_lot':'{:.4f}',
        'theta_per_lot':'{:.4f}',
        'gamma_per_lot':'{:.6f}',
        'vega_per_lot':'{:.4f}',
        'total_delta':'{:.2f}',
        'total_theta':'{:.2f}',
        'total_gamma':'{:.4f}',
        'total_vega':'{:.2f}'
    }), use_container_width=True)

    st.markdown("**Edit table** (edits update stored portfolio):")
    edited = st.experimental_data_editor(df, num_rows="dynamic")
    if not edited.equals(df):
        new_port = []
        for _, r in edited.iterrows():
            new_port.append({
                'expiry_days': int(r['expiry_days']),
                'strike': int(r['strike']),
                'type': r['type'],
                'sigma': float(r['sigma']),
                'lots': int(r['lots']),
                'lot_size': int(r['lot_size']),
                'direction': r['direction']
            })
        st.session_state.portfolio = new_port
        st.experimental_rerun()

    remove_choices = [f"{i} — {df.loc[i,'direction']} {df.loc[i,'type']} {df.loc[i,'strike']} exp {df.loc[i,'expiry_date']}" for i in df.index]
    to_remove = st.multiselect("Select positions to remove", options=list(df.index), format_func=lambda i: remove_choices[i])
    if st.button("Remove selected"):
        st.session_state.portfolio = [p for idx,p in enumerate(st.session_state.portfolio) if idx not in to_remove]
        st.success("Removed selected positions")
        st.experimental_rerun()

    totals = df[['total_delta','total_theta','total_gamma','total_vega']].sum()
    st.subheader("Aggregate Greeks")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Delta", f"{totals.total_delta:.2f}")
    c2.metric("Theta", f"{totals.total_theta:.2f}")
    c3.metric("Gamma", f"{totals.total_gamma:.4f}")
    c4.metric("Vega", f"{totals.total_vega:.2f}")

    st.subheader("Greeks by Expiry")
    by_exp = df.groupby('expiry_days')[['total_delta','total_theta','total_gamma','total_vega']].sum().sort_index()
    st.bar_chart(by_exp)

    st.subheader("Greeks by Option Type")
    by_type = df.groupby('type')[['total_delta','total_theta','total_gamma','total_vega']].sum()
    st.bar_chart(by_type)

    csv_bytes = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_bytes, file_name="options_portfolio_kite.csv", mime="text/csv")
else:
    st.info("Portfolio empty — add positions to compute Greeks.")

# Bottom controls
colA, colB = st.columns(2)
with colA:
    if st.button("Load sample positions"):
        st.session_state.portfolio = [
            {'expiry_days':7,'strike':int(25000),'type':'CE','sigma':0.20,'lots':2,'lot_size':50,'direction':'Long'},
            {'expiry_days':7,'strike':int(25200),'type':'PE','sigma':0.22,'lots':1,'lot_size':50,'direction':'Short'}
        ]
        st.success("Sample positions loaded")
        st.experimental_rerun()
with colB:
    if st.button("Reset portfolio"):
        st.session_state.portfolio = []
        st.success("Portfolio reset")
        st.experimental_rerun()

st.markdown("---")
st.markdown("""
Notes:
- For the guided login: enter your Kite API Key (and Secret) in the sidebar, open the login URL, paste the request_token, and click Get Access Token. Once the token is saved in session, the login UI will be hidden.
- Alternatively paste an existing access_token into the sidebar and the app initializes automatically.
- If you want, I can add:
  - instrument-key lookup helper (shows matching tradingsymbols),
  - websocket real-time updates via KiteTicker to recompute Greeks live,
  - encrypted local storage for the access_token.
Which should I add next?
""")
