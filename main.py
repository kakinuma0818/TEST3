# main.py
# 競馬投資アプリ - Streamlit 完全版（スマホ特化・フル装備）
# 使い方: streamlit run main.py
# requirements: see requirements.txt at repo root

import streamlit as st
import pandas as pd
import numpy as np
import math, itertools, re, io, json
import requests
from bs4 import BeautifulSoup
from datetime import date
from typing import List, Tuple, Dict, Any

# optional integer LP solver
try:
    import pulp
    PULP_AVAILABLE = True
except Exception:
    PULP_AVAILABLE = False

# -------------------------
# UI / styling
# -------------------------
st.set_page_config(page_title="競馬投資アプリ（フル装備）", layout="wide")
ACCENT = "#E07A2C"  # エルメスに近いオレンジ寄り
FONT = "Helvetica, Arial, sans-serif"

st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: {FONT}; }}
/* top layout */
.header-row {{ display:flex; gap:8px; align-items:center; }}
.kv {{ font-size:14px; color:#666; }}
button.stButton>button {{ background-color: {ACCENT}; color: white; border-radius:8px; }}
/* mobile-friendly buttons for horse selection */
.horse-btn {{ 
    display:inline-block; margin:4px; padding:10px 12px; border-radius:8px; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.08); border:1px solid #ddd; 
    font-weight:600;
}}
.horse-btn.selected {{ background:{ACCENT}; color:white; border-color:{ACCENT}; }}
.card {{ background: white; border-radius:12px; padding:12px; margin-bottom:10px; box-shadow:0 2px 6px rgba(0,0,0,0.06); }}
@media (max-width:600px) {{
    .horse-btn {{ padding:14px 16px; font-size:16px; }}
}}
/* fixed table header style */
.table-header {{ font-weight:700; }}
/* small muted text */
.small-muted {{ color:#777; font-size:12px; }}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Constants & defaults
# -------------------------
MIN_UNIT = 100
ALLOW_RATE = 0.90  # allow -10% under target
DEFAULT_BUDGET = 1000
DEFAULT_MUL = 1.5
ADJUST_BLOODLINE_LIST = ["ディープインパクト","キングカメハメハ","ロードカナロア","サンデーサイレンス","キングマンボ","ミスプロ"]

# -------------------------
# Utility: sample race data (fallback)
# -------------------------
def sample_race_df():
    data = {
        "枠":[1,1,2,2,3,3,4,4,5,5],
        "馬番":[1,2,3,4,5,6,7,8,9,10],
        "馬名":["アドマイヤ","カラン","サンプルA","サンプルB","サンプルC","サンプルD","サンプルE","サンプルF","サンプルG","サンプルH"],
        "性齢":["牡4","セ4","牝3","牡5","牡6","牝4","牡3","牝5","牡4","牝4"],
        "斤量":[57,57,54,56,57,55,56,55,57,54],
        "前走体重":[500,502,470,480,488,472,486,474,498,468],
        "脚質":["差し","先行","追込","逃げ","先行","差し","先行","追込","差し","先行"],
        "騎手":["川田","バルザローナ","武豊","福永","横山","池添","ルメール","丹内","田辺","三浦"],
        "調教師":["(栗)藤沢","(美)高木","(栗)池江","(美)友道","(栗)田中","(美)佐藤","(栗)松永","(美)高橋","(栗)音無","(美)岩田"],
        "オッズ":[3.2,5.1,12.5,7.8,20.0,15.0,9.5,30.0,8.0,25.0],
        "人気":[1,2,4,3,6,5,7,10,8,9],
        "血統":["ディープ","キングマンボ","ロード","サンデー","ミスプロ","キングカメ","ディープ","キング","ロード","サンデー"]
    }
    return pd.DataFrame(data)

# -------------------------
# Web scrape helpers (best-effort)
# -------------------------
def fetch_single_odds_netkeiba(race_id: str, timeout:int=6) -> Dict[int,float]:
    if not race_id:
        return {}
    headers = {"User-Agent":"Mozilla/5.0 (compatible)"}
    urls = [
        f"https://race.sp.netkeiba.com/race/odds.html?race_id={race_id}",
        f"https://race.sp.netkeiba.com/race/shutuba.html?race_id={race_id}"
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            odds_map = {}
            # try to find単勝オッズ table rows
            # heuristic: look for td that look like: "1" and "3.2"
            for tr in soup.find_all("tr"):
                cells = [c.get_text(strip=True) for c in tr.find_all(["td","th"])]
                if not cells: continue
                nums = [c for c in cells if re.fullmatch(r"\d{1,2}", c)]
                floats = [c for c in cells if re.fullmatch(r"\d+\.\d+", c)]
                if nums and floats:
                    try:
                        m = int(nums[0]); o = float(floats[0]); odds_map[m]=o
                    except: pass
            if odds_map:
                return odds_map
        except Exception:
            continue
    return {}

def fetch_combo_odds_netkeiba(race_id: str, bet_type: str, timeout:int=6) -> Dict[Tuple,float]:
    # best-effort similar to previous main implementations
    return {}

# -------------------------
# bet expansion
# -------------------------
def expand_combos(bet_type: str, method: str, selections: dict):
    combos=[]
    pool = selections.get('pool', [])
    if bet_type in ('単勝','複勝'):
        return [(h,) for h in pool]
    if bet_type in ('ワイド','馬連','枠連'):
        if method in ('通常','ボックス'):
            combos = list(itertools.combinations(pool,2))
        elif method=='フォーメーション':
            c1=selections.get('col1',[]); c2=selections.get('col2',[])
            for a in c1:
                for b in c2:
                    if a!=b: combos.append(tuple(sorted((a,b))))
        elif method in ('軸1','ながし'):
            axis=selections.get('axis'); ops=selections.get('opponents',[])
            for op in ops: combos.append(tuple(sorted((axis,op))))
        return sorted(set(combos))
    if bet_type=='馬単':
        if method in ('通常','ボックス'):
            combos=list(itertools.permutations(pool,2))
        elif method=='フォーメーション':
            c1=selections.get('col1',[]); c2=selections.get('col2',[])
            for a in c1:
                for b in c2:
                    if a!=b: combos.append((a,b))
        elif method in ('軸1','ながし'):
            axis=selections.get('axis'); ops=selections.get('opponents',[])
            for op in ops: combos.append((axis,op))
        return combos
    if bet_type=='3連複':
        if method in ('通常','ボックス'):
            combos=list(itertools.combinations(pool,3))
        elif method=='フォーメーション':
            c1=selections.get('col1',[]); c2=selections.get('col2',[]); c3=selections.get('col3',[])
            for a in c1:
                for b in c2:
                    for c in c3:
                        if len({a,b,c})==3: combos.append(tuple(sorted((a,b,c))))
        elif method in ('軸1','ながし'):
            axis=selections.get('axis'); ops=selections.get('opponents',[])
            for pair in itertools.combinations(ops,2): combos.append(tuple(sorted((axis,pair[0],pair[1]))))
        elif method=='軸2':
            axes=selections.get('axes',[]); ops=selections.get('opponents',[])
            for op in ops:
                if op not in axes: combos.append(tuple(sorted((*axes,op))))
        return sorted(set(combos))
    if bet_type=='3連単':
        if method in ('通常','ボックス'):
            combos=list(itertools.permutations(pool,3))
        elif method=='フォーメーション':
            c1=selections.get('col1',[]); c2=selections.get('col2',[]); c3=selections.get('col3',[])
            for a in c1:
                for b in c2:
                    for c in c3:
                        if len({a,b,c})==3: combos.append((a,b,c))
        elif method=='軸1':
            axis=selections.get('axis'); ops=selections.get('opponents',[])
            for pair in itertools.permutations(ops,2): combos.append((axis,pair[0],pair[1]))
        elif method=='軸2':
            axes=selections.get('axes',[]); ops=selections.get('opponents',[])
            for op in ops:
                for perm_axes in itertools.permutations(axes,2):
                    if op not in perm_axes: combos.append((perm_axes[0],perm_axes[1],op))
        elif method=='ながし':
            axis=selections.get('axis'); ops=selections.get('opponents',[])
            for pair in itertools.permutations(ops,2): combos.append((axis,pair[0],pair[1]))
        return combos
    return combos

# -------------------------
# combo odds estimator (use single odds or heuristic)
# -------------------------
def estimate_combo_odds_with_lookup(combo, bet_type, single_odds_map, combo_lookup=None):
    if combo_lookup:
        if bet_type in ('馬連','ワイド','枠連','3連複'):
            key = tuple(sorted(combo))
            if key in combo_lookup: return float(combo_lookup[key])
        else:
            key = tuple(combo)
            if key in combo_lookup: return float(combo_lookup[key])
    # fallback: geometric mean * factor
    vals=[float(single_odds_map.get(int(x), 50.0)) for x in combo]
    geo = float(np.exp(np.mean(np.log(np.array(vals)+1e-9))))
    factors={'単勝':1.0,'複勝':0.6,'ワイド':1.5,'馬連':1.6,'馬単':2.0,'3連複':3.0,'3連単':7.0,'枠連':1.4}
    f=factors.get(bet_type,1.6)
    est=round(geo*f,2)
    if est<1.0: est=1.0
    return est

# -------------------------
# scoring module (SC)
# -------------------------
def score_horses(df: pd.DataFrame, surface='芝', focus_bloods=None):
    if focus_bloods is None: focus_bloods = ADJUST_BLOODLINE_LIST
    scores={}
    for _,r in df.iterrows():
        m=int(r['馬番'])
        total=0
        subs={}
        # age
        age_match=re.search(r'(\d+)', str(r.get('性齢','')))
        age=int(age_match.group(1)) if age_match else 4
        if surface=='ダ':
            a = 3.0 if age in (3,4) else 2.0 if age==5 else 1.5 if age==6 else 1.0
        else:
            a = 3.0 if age in (3,4,5) else 2.0 if age==6 else 1.0
        subs['年齢']=a; total+=a
        # blood
        blood=str(r.get('血統',''))
        b=1.0
        for fb in focus_bloods:
            if fb in blood: b+=0.5
        if any(s in blood for s in ["ディープ","キング","ロード","サンデー","ミスプロ"]): b+=0.5
        subs['血統']=b; total+=b
        # jockey placeholder
        j = 2.0
        subs['騎手']=j; total+=j
        # trainer
        subs['調教師']=1.5; total+=1.5
        # owner breeder placeholders
        subs['馬主']=1.0; total+=1.0
        subs['生産者']=1.0; total+=1.0
        # form via 人気
        rank=int(r.get('人気',10))
        f=3.0 if rank<=3 else 2.0 if rank<=6 else 1.0
        subs['成績']=f; total+=f
        # distance/track neutral
        subs['競馬場']=1.5; total+=1.5
        subs['距離']=1.5; total+=1.5
        # style
        style_map={'逃げ':3.0,'先行':2.5,'差し':2.0,'追込':1.5}
        s=style_map.get(str(r.get('脚質','')),2.0)
        subs['脚質']=s; total+=s
        #枠補正
        subs['枠']=1.0; total+=1.0
        #馬場補正: default 0 (applied later)
        subs['馬場']=0.0; total+=0.0
        subs['合計']=round(total,2)
        scores[m]=subs
    return scores

# -------------------------
# Optimization / allocation
# -------------------------
def compute_payouts(bet_list, combo_odds):
    return [int(round(a * o)) for a,o in zip(bet_list, combo_odds)]

def allocate_with_lp(total_budget, target_mul, combo_odds, allow_rate=ALLOW_RATE, time_limit_sec=5):
    n=len(combo_odds)
    if n==0: return {"ok":False,"error":"no_combos"}
    target_return = float(total_budget)*float(target_mul)
    min_return = target_return*allow_rate
    if not PULP_AVAILABLE:
        return {"ok":False,"error":"pulp_missing"}
    prob = pulp.LpProblem("alloc", pulp.LpMinimize)
    x_vars=[pulp.LpVariable(f"x_{i}", lowBound=0, cat='Integer') for i in range(n)]
    prob += pulp.lpSum([x_vars[i] for i in range(n)])
    prob += pulp.lpSum([MIN_UNIT * x_vars[i] for i in range(n)]) <= total_budget
    slack=[pulp.LpVariable(f"s_{i}", lowBound=0, cat='Integer') for i in range(n)]
    BIG=10**7
    for i in range(n):
        prob += combo_odds[i] * (MIN_UNIT * x_vars[i]) + BIG*slack[i] >= min_return
    prob += pulp.lpSum([BIG * slack[i] for i in range(n)]) * 0.000001
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_sec)
    try:
        prob.solve(solver)
    except Exception:
        return {"ok":False,"error":"lp_failed"}
    if pulp.LpStatus.get(prob.status,None)=='Optimal' or prob.status==1:
        xvals=[int(pulp.value(x_vars[i])) for i in range(n)]
        allocation=[int(MIN_UNIT * x) for x in xvals]
        if sum(allocation) <= total_budget and any(a>0 for a in allocation):
            return {"ok":True,"alloc":allocation}
        else:
            return {"ok":False,"error":"lp_infeasible_or_trivial"}
    return {"ok":False,"error":"lp_no_opt"}

def calc_allocations_greedy(total_budget, target_mul, combo_odds, allow_rate=ALLOW_RATE):
    n=len(combo_odds)
    if n==0: return {"ok":False,"error":"no_combos"}
    target_return = float(total_budget)*float(target_mul)
    min_return = target_return*allow_rate
    required_for_target=[]
    for o in combo_odds:
        if o<=0: need_100=float('inf')
        else:
            need = target_return / float(o)
            need_100 = int(math.ceil(need / MIN_UNIT) * MIN_UNIT)
        required_for_target.append(need_100)
    if sum(required_for_target) <= total_budget:
        allocation = required_for_target.copy()
        rem = total_budget - sum(allocation)
        idxs = sorted(range(n), key=lambda i: combo_odds[i], reverse=True)
        i=0
        while rem >= MIN_UNIT:
            allocation[idxs[i % n]] += MIN_UNIT
            rem -= MIN_UNIT
            i += 1
        return {"ok":True,"method":"target_exact","bet_list":allocation}
    required_for_min=[]
    for o in combo_odds:
        if o<=0: need_100=float('inf')
        else:
            need = min_return / float(o)
            need_100 = int(math.ceil(need / MIN_UNIT) * MIN_UNIT)
        required_for_min.append(need_100)
    if sum(required_for_min) <= total_budget:
        allocation = required_for_min.copy()
        rem = total_budget - sum(allocation)
        while rem >= MIN_UNIT:
            gaps = [max(0.0, target_return - (combo_odds[i] * allocation[i])) for i in range(n)]
            if sum(gaps)==0:
                idxs = sorted(range(n), key=lambda i: combo_odds[i], reverse=True)
                allocation[idxs[0]] += MIN_UNIT
                rem -= MIN_UNIT
                continue
            idx = int(np.argmax(gaps))
            allocation[idx] += MIN_UNIT
            rem -= MIN_UNIT
        return {"ok":True,"method":"min_allowed","bet_list":allocation}
    required_budget = sum(required_for_min)
    recommended_mul=None
    for mul in np.arange(target_mul, 0.99, -0.5):
        tr = float(total_budget)*float(mul)
        minr = tr*allow_rate
        needed = [int(math.ceil((minr / o) / MIN_UNIT) * MIN_UNIT) if o>0 else float('inf') for o in combo_odds]
        if sum(needed) <= total_budget:
            recommended_mul = mul
            break
    return {"ok":False,"mode":"impossible","required_budget":required_budget,"recommended_mul":recommended_mul}

def allocate(total_budget, target_mul, combo_odds, allow_rate=ALLOW_RATE):
    if PULP_AVAILABLE:
        res = allocate_with_lp(total_budget, target_mul, combo_odds, allow_rate)
        if res.get('ok'): return {"ok":True,"method":"lp","allocation":res['alloc']}
    res_g = calc_allocations_greedy(total_budget, target_mul, combo_odds, allow_rate)
    return res_g

# -------------------------
# helper: CSV download
# -------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding='utf-8-sig')
    return buf.getvalue()

# -------------------------
# Session init
# -------------------------
if 'fetched_odds' not in st.session_state: st.session_state['fetched_odds'] = {}
if 'fetched_combo_odds' not in st.session_state: st.session_state['fetched_combo_odds'] = {}
if 'combo_info' not in st.session_state: st.session_state['combo_info'] = None
if 'last_alloc' not in st.session_state: st.session_state['last_alloc'] = None

# -------------------------
# Top controls: race selector + investment
# -------------------------
st.title("競馬投資アプリ — フル装備（スマホ向け）")

top_cols = st.columns([2,2,2,1,1])
race_date = top_cols[0].date_input("開催日", value=date.today())
track = top_cols[1].selectbox("競馬場", ["東京","中山","京都","阪神","新潟","福島","札幌","函館"], index=0)
race_no = top_cols[2].selectbox("レース番号", [f"{i}R" for i in range(1,13)], index=10)
race_id = top_cols[3].text_input("race_id (任意)", value="")
if top_cols[4].button("オッズ取得"):
    st.info("オッズ取得を試みます（netkeiba）。失敗時は内部推定にフォールバック")
    single = fetch_single_odds_netkeiba(race_id)
    if single:
        st.success("単勝オッズを取得しました")
        st.session_state['fetched_odds'] = single
    else:
        st.warning("オッズ取得失敗：推定オッズを使います")

# investment settings (sidebar-like but in top area for mobile)
inv_col1, inv_col2, inv_col3 = st.columns([1,1,1])
total_budget = inv_col1.number_input("総投資額 (円)", min_value=100, step=100, value=DEFAULT_BUDGET)
mul_choices = [i/2 for i in range(2,41)]
target_mul = inv_col2.selectbox("希望払い戻し倍率", mul_choices, index=mul_choices.index(DEFAULT_MUL))
allow_slider = inv_col3.slider("許容下振れ (%)", 0, 30, int((1-ALLOW_RATE)*100), step=1)
ALLOW_RATE = 1 - (allow_slider/100)

# load race data (sample for now)
df = sample_race_df()
# override odds if fetched
if st.session_state.get('fetched_odds'):
    sm = st.session_state['fetched_odds']
    for idx,row in df.iterrows():
        b = int(row['馬番'])
        if b in sm:
            df.at[idx,'オッズ'] = sm[b]

# compute scores
scores = score_horses(df, surface='芝')

# -------------------------
# MA / tabs: display
# -------------------------
tabs = st.tabs(["出馬表","SC","成績","PR","馬券"])
# -- 出馬表 --
with tabs[0]:
    st.markdown("### 出馬表")
    st.markdown("タップで馬を選択（買い目作成は馬券タブで）")
    # render horse buttons grid (mobile friendly)
    cols_count = 4
    btn_cols = st.container()
    with btn_cols:
        for i,row in df.iterrows():
            bnum = int(row['馬番'])
            name = row['馬名']
            odds = row['オッズ']
            pop = row['人気']
            key = f"hb_{bnum}"
            if key not in st.session_state:
                st.session_state[key] = False
            btn_label = f"{bnum}\n{name}\n{odds}倍\n({pop})"
            css = "horse-btn selected" if st.session_state[key] else "horse-btn"
            # use button to toggle selection
            if st.button(btn_label, key=f"btn_{bnum}"):
                st.session_state[key] = not st.session_state[key]
    # small legend
    st.markdown("<div class='small-muted'>馬名をタップすると選択されます。馬券タブで選択済み馬を反映します。</div>", unsafe_allow_html=True)

# -- SC tab --
with tabs[1]:
    st.markdown("### SC（スコア）")
    st.markdown("馬名と合計を左に固定。トップ3は太字表示。手動補正（-3〜+3）を各馬に適用可。")
    # build DataFrame for display
    rows=[]
    for _,r in df.iterrows():
        b=int(r['馬番'])
        sc = scores[b]
        manual_key = f"manual_{b}"
        if manual_key not in st.session_state:
            st.session_state[manual_key] = 0
        rows.append({
            "馬番":b,
            "馬名":r['馬名'],
            "合計": sc['合計'] + st.session_state[manual_key],
            "年齢":sc['年齢'],
            "血統":sc['血統'],
            "騎手":sc['騎手'],
            "調教師":sc['調教師'],
            "成績":sc['成績'],
            "競馬場":sc['競馬場'],
            "距離":sc['距離'],
            "脚質":sc['脚質'],
            "枠":sc['枠'],
            "馬場":sc['馬場'],
            "手動":st.session_state[manual_key]
        })
    sc_df = pd.DataFrame(rows).sort_values(by="合計", ascending=False)
    # bold top3
    def highlight_top3(val, rank_dict):
        # used in display later if needed
        return val
    st.dataframe(sc_df.reset_index(drop=True), use_container_width=True)
    # manual adjustments area
    st.markdown("**手動補正（-3〜+3）：**")
    for _,r in df.iterrows():
        b=int(r['馬番'])
        st.slider(f"{b}. {r['馬名']}", -3, 3, st.session_state.get(f"manual_{b}",0), key=f"manual_{b}", help="スコアに加算されます")

# -- 成績 tab --
with tabs[2]:
    st.markdown("### 成績（過去5戦）")
    st.markdown("デモ：各馬の直近5戦の着順・時計・上り・体重を表示（データ接続で実データに差替可能）")
    if st.button("過去成績を生成（デモ）"):
        rec_rows=[]
        for _,r in df.iterrows():
            for i in range(1,6):
                rec_rows.append({
                    "馬番":r['馬番'],
                    "馬名":r['馬名'],
                    "日付":f"2024-0{i}-0{i}",
                    "レース名":f"デモ{i}",
                    "距離":"1800m",
                    "馬場":"良",
                    "着順": np.random.randint(1,12),
                    "走破時計": f"{120 + np.random.randint(0,20)}.0",
                    "上り": f"{34 + np.random.randint(0,5)}.0",
                    "体重": r['前走体重'] + np.random.randint(-6,6)
                })
        rec_df = pd.DataFrame(rec_rows)
        st.dataframe(rec_df, use_container_width=True)
    else:
        st.info("表示ボタンを押すとデモの過去成績を生成します")

# -- PR tab --
with tabs[3]:
    st.markdown("### 基本情報（PR）")
    horse_choice = st.selectbox("馬を選択", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
    if horse_choice:
        num=int(horse_choice.split(".")[0])
        r=df[df['馬番']==num].iloc[0]
        st.markdown(f"**{r['馬名']}**")
        st.write(f"性齢: {r['性齢']}　斤量: {r['斤量']}　前走体重: {r['前走体重']}")
        st.write(f"騎手: {r['騎手']}　調教師: {r['調教師']}")
        st.write(f"血統（5代は非表示）: {r['血統']}")
        st.write("馬主: (データ接続で表示)  生産者: (データ接続で表示)")
        st.write("※調教師は（栗/美）表記済")

# -- 馬券 tab (BE) --
with tabs[4]:
    st.markdown("### 馬券（Betting） — フロー: 種類→買い方→馬選択→点数確認→自動配分")
    bet_types = ["単勝","複勝","ワイド","馬連","馬単","3連複","3連単","枠連"]
    bet_type = st.selectbox("馬券種", bet_types)
    # methods UI
    if bet_type in ["単勝","複勝"]:
        method = "通常"
    elif bet_type in ["ワイド","馬連","枠連"]:
        method = st.selectbox("買い方", ["通常","ボックス","フォーメーション","軸1","ながし"])
    elif bet_type == "馬単":
        method = st.selectbox("買い方", ["通常","ボックス","フォーメーション","軸1","ながし","マルチ"])
    else:
        method = st.selectbox("買い方", ["通常","ボックス","フォーメーション","軸1","軸2","ながし"])

    selections={}
    if method=="フォーメーション":
        if bet_type in ["馬連","馬単","ワイド","枠連"]:
            col1 = st.multiselect("1列目", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            col2 = st.multiselect("2列目", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            selections['col1']=[int(x.split(".")[0]) for x in col1]; selections['col2']=[int(x.split(".")[0]) for x in col2]
        else:
            col1=st.multiselect("1列目", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            col2=st.multiselect("2列目", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            col3=st.multiselect("3列目", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            selections['col1']=[int(x.split(".")[0]) for x in col1]; selections['col2']=[int(x.split(".")[0]) for x in col2]; selections['col3']=[int(x.split(".")[0]) for x in col3]
    elif method in ("軸1","軸2","ながし"):
        axis = st.selectbox("軸を選択", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
        opponents = st.multiselect("相手を選択", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
        selections['axis']=int(axis.split(".")[0]); selections['opponents']=[int(x.split(".")[0]) for x in opponents]
    else:
        # default pool is from selected buttons if any; else manual multi-select
        pool=[]
        for _,r in df.iterrows():
            if st.session_state.get(f"hb_{int(r['馬番'])}", False):
                pool.append(int(r['馬番']))
        if not pool:
            pool = st.multiselect("購入馬（ボタン未使用の場合はここで選択）", [f"{int(r['馬番'])}. {r['馬名']}" for _,r in df.iterrows()])
            pool=[int(x.split(".")[0]) for x in pool]
        selections['pool'] = pool

    # show estimated points
    if st.button("点数計算・買い目生成"):
        combos = expand_combos(bet_type, method, selections)
        if not combos:
            st.warning("買い目が生成されませんでした。選択を確認してください。")
        else:
            single_map = {int(r['馬番']): float(r['オッズ']) for _,r in df.iterrows()}
            combo_lookup = st.session_state.get('fetched_combo_odds', {})
            combo_lookup_for_type = combo_lookup.get(bet_type, {}) if combo_lookup else {}
            combo_desc=[]
            combo_odds=[]
            combo_scores=[]
            for c in combos:
                desc = "-".join(map(str,c)) if bet_type not in ['馬単','3連単'] else ">".join(map(str,c))
                o = estimate_combo_odds_with_lookup(c, bet_type, single_map, combo_lookup_for_type)
                combo_desc.append(desc); combo_odds.append(o)
                combo_scores.append(sum([scores.get(int(x),{}).get('合計',0) for x in c]))
            combo_df = pd.DataFrame({"買い目":combo_desc,"目安オッズ":combo_odds,"スコア合計":combo_scores})
            st.success(f"買い目生成完了：{len(combos)} 点")
            st.dataframe(combo_df, use_container_width=True)
            st.session_state['combo_info'] = {"combos":combos,"desc":combo_desc,"odds":combo_odds,"scores":combo_scores,"bet_type":bet_type}
    # allocation zone
    st.markdown("---")
    st.subheader("自動配分（100円刻み）")
    if st.session_state.get('combo_info'):
        info = st.session_state['combo_info']
        combos = info['combos']; descs = info['desc']; odds = info['odds']; scores_list = info['scores']
        st.write(f"買い目数: {len(combos)}  /  目標払戻: {int(total_budget * target_mul):,} 円  / 許容下限: {int(total_budget * target_mul * ALLOW_RATE):,} 円")
        if st.button("自動配分を実行"):
            res = allocate(total_budget, target_mul, odds, ALLOW_RATE)
            if not res.get('ok'):
                st.error("自動配分できませんでした（投資不足・最適化失敗）")
                if res.get('error')=='pulp_missing':
                    st.info("最適化ライブラリ(pulp)が無効です。貪欲法を試します。")
                    res = calc_allocations_greedy(total_budget, target_mul, odds, ALLOW_RATE)
                if not res.get('ok'):
                    st.write("必要最低投資額:", res.get('required_budget'))
                    if res.get('recommended_mul'): st.write("推奨倍率:", res.get('recommended_mul'))
            if res.get('ok'):
                allocation = res.get('allocation') if res.get('method')=='lp' else res.get('bet_list')
                if isinstance(allocation, list):
                    payouts = compute_payouts(allocation, odds)
                    rows=[]
                    for d,o,a,p,sc in zip(descs, odds, allocation, payouts, scores_list):
                        rows.append({"買い目":d,"目安オッズ":f"{o:.2f}","推奨金額(円)":a,"期待払戻(円)":p,"期待倍率": round(p/max(1,a),2) if a>0 else 0,"スコア合計":sc})
                    df_alloc = pd.DataFrame(rows)
                    st.success("自動配分完了")
                    st.dataframe(df_alloc, use_container_width=True)
                    st.session_state['last_alloc'] = {"alloc": allocation, "df": df_alloc}
                else:
                    st.error("配分結果の形式が不正です")
        # manual adjust
        if st.session_state.get('last_alloc'):
            st.markdown("**自動配分結果（手動微調整可）**")
            alloc = st.session_state['last_alloc']['alloc'].copy()
            edited=[]; total_now=0
            for i,(d,o,a) in enumerate(zip(descs, odds, alloc)):
                cols = st.columns([4,2,2])
                cols[0].write(d)
                cols[1].write(f"{o:.2f}")
                new_amt = cols[2].number_input(f"金額編集 {i}", min_value=0, step=MIN_UNIT, value=int(a), key=f"edit_{i}")
                edited.append(new_amt); total_now += new_amt
            st.markdown(f"合計投資額（現在）: {total_now:,} 円 / 設定総投資額: {total_budget:,} 円")
            if total_now > total_budget:
                st.warning("合計投資額が総投資を超えています。調整してください。")
            else:
                st.success("合計は総投資額内です。")
            if st.button("編集結果を確定（保存）"):
                st.session_state['manual_alloc'] = {"combos":combos,"desc":descs,"odds":odds,"alloc":edited}
                st.success("編集結果を保存しました")
                # CSV download button
                out_df = pd.DataFrame({"買い目":descs,"目安オッズ":odds,"投票額":edited})
                csv_bytes = df_to_csv_bytes(out_df)
                st.download_button("CSVをダウンロード", data=csv_bytes, file_name="bets.csv", mime="text/csv")
    else:
        st.info("まずは買い目を生成してください")

# Footer notes
st.markdown("---")
st.markdown("*注: 組合せオッズは推定または取得した目安を使用しています。実際の購入はJRA等公式サイトで行ってください。*")
if not PULP_AVAILABLE:
    st.info("整数最適化 (pulp) がインストールされていません。LP最適化は使えませんが、貪欲法で配分します。")

# End of app
