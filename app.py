# -*- coding: utf-8 -*-
"""
古风网络语言语料库 · Streamlit 交互界面（云端部署版）
功能：数据上传与管理、多维筛选检索、统计指标、词频/词云、语义网络、结果导出。
路径全部基于脚本所在目录，字体使用自带开源字体 Noto Sans SC (OFL)。
"""
import os
import re
from collections import Counter

import jieba
import jieba.posseg as pseg
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go

# ---------- 基础配置 ----------
st.set_page_config(page_title="古风网络语言语料库", page_icon="📜", layout="wide")

# 路径：基于脚本所在目录（本地 / 云端通用）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(BASE_DIR, "corpus.csv")
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NotoSansSC.otf")

# 注册自带中文字体（Noto Sans SC）
font_manager.fontManager.addfont(FONT_PATH)
FONT_NAME = font_manager.FontProperties(fname=FONT_PATH).get_name()
matplotlib.rcParams["font.sans-serif"] = [FONT_NAME, "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

DIM_ORDER = ["古风诗词语素拼接类", "网络化诗词改编类", "古风原词使用类"]
DIM_COLOR = {"古风诗词语素拼接类": "#8C6A5D",
             "网络化诗词改编类": "#C9A14A",
             "古风原词使用类": "#5E7A6E"}

PUNCT = re.compile(r"[，。？！；：、“”‘’「」《》（）【】…—·,.\?\!;:\(\)\[\]\"'~`@#\$%\^&\*\-_=\+\|/<>\\\s]")
USER_WORDS = ["小女子不才", "小生不才", "公子", "姑娘", "少侠", "在下", "噫吁嚱", "快哉",
              "本宫", "极好", "恩泽", "小主", "方才", "私心", "想着", "夫君", "先生",
              "莫怪", "莫慌", "奈何", "罢了"]
STOP = set("的 了 是 我 你 他 她 它 们 着 就 都 也 在 和 与 及 啊 呀 吗 呢 吧 哦 一个 一 个 这 那 有 不 把 被 让 给 还 又 很 却 便 即 若 如 为 以 于 之 其 而".split())


def sen_len(line):
    """句长口径：只数汉字、数字、外文词，不含标点。"""
    return len(re.findall(r"[\u4E00-\u9FFF]|-?\d+(\.\d+)?|[a-zA-Z]+", str(line)))


# ---------- 数据加载 ----------
@st.cache_data(show_spinner=False)
def load_default():
    try:
        return pd.read_csv(DEFAULT_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=["内容", "类型", "子类", "平台", "维度"])


@st.cache_data(show_spinner=False)
def parse_upload(raw):
    return pd.read_csv(raw)


def get_data():
    """侧边栏选择数据源：内置库或上传。"""
    st.sidebar.subheader("📁 数据源")
    src = st.sidebar.radio("选择数据源", ["内置语料库", "上传 CSV"], label_visibility="collapsed")
    if src == "上传 CSV":
        up = st.sidebar.file_uploader("上传三维分类 CSV（须含“内容”列）", type=["csv"])
        if up is not None:
            df = parse_upload(up)
            st.sidebar.success(f"已加载：{up.name}（{len(df)} 条）")
            return df
        st.sidebar.info("未检测到文件，暂用内置库。")
    return load_default()


# ---------- 分析函数 ----------
def init_jieba(df):
    """加载自定义词与全部 X而Y之 构式（含字母/数字构式）。"""
    for w in USER_WORDS:
        jieba.add_word(w)
    if "子类" in df.columns:
        for _, r in df.iterrows():
            if str(r["子类"]).startswith("X而Y之") or r["子类"] == "古成语":
                jieba.add_word(PUNCT.sub("", str(r["内容"])))


@st.cache_data(show_spinner=False)
def tokenize_texts(tuple_texts):
    """分词并返回词列表（须先调用 init_jieba）。"""
    words = []
    for t in tuple_texts:
        for w, fl in pseg.cut(str(t)):
            w = PUNCT.sub("", w)
            if w:
                words.append(w)
    return words


def metrics_block(df):
    texts = tuple(df["内容"].astype(str))
    words = tokenize_texts(texts)
    n = len(df)
    nchar = int(sum(sen_len(t) for t in df["内容"].astype(str)))
    cols = st.columns(4)
    cols[0].metric("语料条数", f"{n:,}")
    cols[1].metric("总形符", f"{len(words):,}")
    cols[2].metric("类符 / TTR", f"{len(set(words)):,} / {len(set(words))/len(words):.3f}" if words else "-")
    cols[3].metric("平均句长", f"{nchar/n:.2f} 字" if n else "-")
    return words


# ---------- 页面：概览 ----------
def page_overview(df):
    st.title("📊 语料库概览")
    metrics_block(df)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("三维分类占比")
        if "维度" in df.columns:
            d = df["维度"].value_counts().reset_index()
            d.columns = ["维度", "条数"]
            d["维度"] = pd.Categorical(d["维度"], DIM_ORDER, ordered=True)
            d = d.sort_values("维度")
            fig = px.pie(d, names="维度", values="条数", hole=0.4,
                         color="维度", color_discrete_map=DIM_COLOR)
            fig.update_traces(textinfo="percent+value")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("类型分布")
        if "类型" in df.columns:
            t = df["类型"].value_counts().reset_index()
            t.columns = ["类型", "条数"]
            fig = px.bar(t, x="条数", y="类型", orientation="h",
                         color="类型", color_discrete_sequence=px.colors.qualitative.T10)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("平台 / 子类分布")
    c3, c4 = st.columns(2)
    with c3:
        if "平台" in df.columns:
            p = df["平台"].value_counts().reset_index()
            p.columns = ["平台", "条数"]
            fig = px.bar(p, x="平台", y="条数", color="平台",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    with c4:
        if "子类" in df.columns:
            s = df["子类"].value_counts().head(12).reset_index()
            s.columns = ["子类", "条数"]
            fig = px.bar(s, x="条数", y="子类", orientation="h",
                         color="子类", color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


# ---------- 页面：检索 ----------
def page_search(df):
    st.title("🔍 多维检索")
    filtered = df.copy()

    cols = st.columns(4)
    if "维度" in df.columns:
        v = cols[0].multiselect("维度", sorted(df["维度"].dropna().unique()))
        if v:
            filtered = filtered[filtered["维度"].isin(v)]
    if "类型" in df.columns:
        t = cols[1].multiselect("类型", sorted(df["类型"].dropna().unique()))
        if t:
            filtered = filtered[filtered["类型"].isin(t)]
    if "子类" in df.columns:
        sub = cols[2].multiselect("子类", sorted(df["子类"].dropna().unique()))
        if sub:
            filtered = filtered[filtered["子类"].isin(sub)]
    if "平台" in df.columns:
        p = cols[3].multiselect("平台", sorted(df["平台"].dropna().unique()))
        if p:
            filtered = filtered[filtered["平台"].isin(p)]

    kw = st.text_input("关键词搜索（内容包含）", "")
    if kw.strip():
        filtered = filtered[filtered["内容"].astype(str).str.contains(kw.strip(), regex=False)]

    st.caption(f"命中 {len(filtered)} 条")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=480)

    if len(filtered):
        st.download_button("⬇️ 导出检索结果 CSV",
                           filtered.to_csv(index=False).encode("utf-8-sig"),
                           "检索结果.csv", "text/csv")


# ---------- 页面：词频 ----------
def page_freq(df):
    st.title("📈 词频分析")
    scope = st.selectbox("分析范围", ["全库"] + (DIM_ORDER if "维度" in df.columns else []))
    sub = df if scope == "全库" else df[df["维度"] == scope]
    topn = st.slider("TOP N", 10, 50, 20)
    use_stop = st.checkbox("去停用词", True)

    words = tokenize_texts(tuple(sub["内容"].astype(str)))
    if use_stop:
        words = [w for w in words if w not in STOP]
    fw = Counter(words).most_common(topn)

    chars = [ch for t in sub["内容"].astype(str) for ch in PUNCT.sub("", str(t))]
    fc = Counter(chars).most_common(topn)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"高频词 TOP{topn}")
        if fw:
            d = pd.DataFrame(fw, columns=["词", "频次"]).iloc[::-1]
            fig = px.bar(d, x="频次", y="词", orientation="h",
                         color_discrete_sequence=["#8C6A5D"])
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader(f"高频字 TOP{topn}")
        if fc:
            d = pd.DataFrame(fc, columns=["字", "频次"]).iloc[::-1]
            fig = px.bar(d, x="频次", y="字", orientation="h",
                         color_discrete_sequence=["#5E7A6E"])
            st.plotly_chart(fig, use_container_width=True)


# ---------- 页面：词云 ----------
def page_cloud(df):
    st.title("☁️ 词云")
    scope = st.selectbox("词云范围", ["全库"] + (DIM_ORDER if "维度" in df.columns else []))
    sub = df if scope == "全库" else df[df["维度"] == scope]
    use_stop = st.checkbox("去停用词", True)
    if st.button("🎨 生成词云"):
        words = tokenize_texts(tuple(sub["内容"].astype(str)))
        if use_stop:
            words = [w for w in words if w not in STOP]
        wc = WordCloud(font_path=FONT_PATH, width=1200, height=700,
                       background_color="white",
                       color_func=lambda *a, **k: "#8C6A5D").generate(" ".join(words))
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)


# ---------- 页面：语义网络 ----------
def page_network(df):
    st.title("🕸️ 词语共现网络")
    scope = st.selectbox("网络范围", ["全库"] + (DIM_ORDER if "维度" in df.columns else []))
    sub = df if scope == "全库" else df[df["维度"] == scope]
    fthr = st.slider("节点最低频次", 5, 50, 20)
    cthr = st.slider("最低共现次数", 2, 10, 3)

    words = [w for w in tokenize_texts(tuple(sub["内容"].astype(str))) if w not in STOP]
    fw = Counter(words)
    nodes = {w for w, n in fw.items() if n >= fthr}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for t in sub["内容"].astype(str):
        ws = [w for w in tokenize_texts((t,)) if w in nodes]
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                if ws[i] != ws[j]:
                    e = tuple(sorted((ws[i], ws[j])))
                    if G.has_edge(*e):
                        G[e[0]][e[1]]["weight"] += 1
                    else:
                        G.add_edge(e[0], e[1], weight=1)
    G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d["weight"] < cthr])
    G.remove_nodes_from(list(nx.isolates(G)))

    st.caption(f"节点 {G.number_of_nodes()} ｜ 边 {G.number_of_edges()}")
    if G.number_of_nodes() == 0:
        st.info("当前阈值下无网络，请降低阈值。")
        return

    pos = nx.spring_layout(G, k=0.5, seed=42)
    deg = dict(G.degree())
    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                               line=dict(width=0.6, color="#bbb"), hoverinfo="none"))
    nx_ = [n for n in G.nodes()]
    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in nx_], y=[pos[n][1] for n in nx_],
        mode="markers+text", text=nx_, textposition="top center",
        marker=dict(size=[8 + deg[n] for n in nx_], color="#8C6A5D",
                    line=dict(width=1, color="white")),
        textfont=dict(size=11), hovertemplate="%{text}（度%{marker.size}）<extra></extra>"))
    fig.update_layout(showlegend=False, height=650, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ---------- 页面：数据管理 ----------
def page_manage(df):
    st.title("🗂️ 数据管理")
    st.subheader("当前数据字段")
    st.write(pd.DataFrame({"字段": df.columns,
                           "非空数": [df[c].notna().sum() for c in df.columns],
                           "示例": [str(df[c].dropna().iloc[0]) if df[c].notna().any() else "" for c in df.columns]}))

    st.subheader("各字段取值分布")
    col = st.selectbox("选择字段", [c for c in df.columns if c != "内容"])
    st.write(df[col].value_counts().reset_index().rename(columns={"index": col, col: "条数"}))

    st.divider()
    st.subheader("上传新数据合并")
    up = st.file_uploader("选择要合并的 CSV", type=["csv"], key="merge")
    if up is not None:
        new = pd.read_csv(up)
        st.write(f"新数据 {len(new)} 条，字段：{list(new.columns)}")
        if "内容" in new.columns:
            combined = pd.concat([df, new], ignore_index=True).drop_duplicates(subset=["内容"])
            st.write(f"合并去重后 {len(combined)} 条")
            st.download_button("⬇️ 下载合并后 CSV",
                               combined.to_csv(index=False).encode("utf-8-sig"),
                               "合并语料库.csv", "text/csv")


# ---------- 主流程 ----------
def main():
    st.sidebar.title("📜 古风网语语料库")
    df = get_data()
    if df.empty or "内容" not in df.columns:
        st.warning("未加载到有效数据，请在左侧上传含“内容”列的 CSV。")
        return
    init_jieba(df)

    page = st.sidebar.radio("功能导航",
                            ["📊 概览", "🔍 多维检索", "📈 词频分析",
                             "☁️ 词云", "🕸️ 语义网络", "🗂️ 数据管理"])
    st.sidebar.caption(f"当前共 {len(df):,} 条")

    {"📊 概览": page_overview, "🔍 多维检索": page_search,
     "📈 词频分析": page_freq, "☁️ 词云": page_cloud,
     "🕸️ 语义网络": page_network, "🗂️ 数据管理": page_manage}[page](df)


if __name__ == "__main__":
    main()
