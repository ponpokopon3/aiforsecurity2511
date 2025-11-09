#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Streamlit frontend for main2.py (cleaned header)

from pathlib import Path
import streamlit as st
import json
import re

try:
    import main as backend
except Exception as e:
    st.error("バックエンド main2.py の読み込みに失敗しました。先に main2.py が正しく動くか確認してください。")
    st.exception(e)
    raise

st.set_page_config(page_title="AgentRAG - Web UI", layout="wide")
st.title("AgentRAG — セキュリティ規則チェッカー (Streamlit UI)")

# 小さめフォントとコンパクト表示のための簡易 CSS
st.markdown(
    """
    <style>
    * { font-size:13px !important; }
    .stButton>button { padding:4px 8px !important; font-size:13px !important; }
    textarea { font-size:12px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# キャッシュ付き初期化
@st.cache_resource
def get_vectordb():
    docs = backend.load_spec_documents(backend.SPEC_DIR)
    return backend.init_chroma(docs)

@st.cache_resource
def get_llm():
    return backend.make_chat_model()

vectordb = None
llm = None
try:
    vectordb = get_vectordb()
    llm = get_llm()
except Exception as e:
    st.warning("ベクトルDB や LLM の初期化で警告が出ました。OpenAIキーや依存が正しく設定されているか確認してください。")
    st.exception(e)

# サイドバー: ページ切替と共通オプション
rule_files = list((Path(__file__).parent / "rule").rglob("*.json"))
st.sidebar.write(f"読み込みルール数: {len(rule_files)}")
page = st.sidebar.radio("ページ", ["ルールチェック", "RAG 質問"])
topk = st.sidebar.slider("RAG: 参照ドキュメント数 (k)", 1, 10, backend.TOP_K)

# ルール一覧を取得
rules = backend.load_rules_from_dir(backend.RULE_DIR)
rule_map = {f"{r.get('id') or ''} | {r.get('title')}": r for r in rules}

if page == "ルールチェック":
    st.header("ルールチェック")
    choices = ["(選択してください)"] + list(rule_map.keys())
    sel = st.selectbox("評価するルールを選択", choices)
    st.caption("ルールを選択して 'チェック実行 (A->B)' を押すと評価が始まります。")

    if sel == "(選択してください)":
        st.info("ルールを選択してください")
    else:
        r = rule_map[sel]
        st.markdown(f"**id**: `{r.get('id')}`  ")
        st.markdown(f"**title**: {r.get('title')}")
        st.write(r.get('content'))

        if st.button("チェック実行"):
            try:
                docs = backend.retrieve_related_docs(vectordb, r.get('content') or r.get('title') or "", k=topk)
                st.write(f"取得ドキュメント: {len(docs)} チャンク（上位 {topk}）")
                st.info("要約中...")
                summary = backend.agent_a_summarize(llm, r.get('content') or '', docs)
                st.success("要約完了")
                # Streamlit 上では要約のプレビューは不要

                st.info("評価中...")
                b_result = backend.agent_b_check(llm, summary, r.get('raw', {}), docs)
                st.success("評価完了")
                st.subheader("判定（Agent B）")
                b_text = backend.format_b_result(b_result)
                # 表示用に改行や余分な空行を整形
                def _normalize_display(text: str) -> str:
                    if not text:
                        return ""
                    t = text.replace('\r\n', '\n').replace('\r', '\n')
                    t = re.sub(r"\n{3,}", "\n\n", t)
                    lines = [ln.rstrip() for ln in t.split('\n')]
                    while lines and lines[0].strip() == "":
                        lines.pop(0)
                    while lines and lines[-1].strip() == "":
                        lines.pop()
                    out_lines = []
                    prev_blank = False
                    for ln in lines:
                        if ln.strip() == "":
                            if not prev_blank:
                                out_lines.append("")
                            prev_blank = True
                        else:
                            out_lines.append(ln.lstrip())
                            prev_blank = False
                    return "\n".join(out_lines)

                b_text_clean = _normalize_display(b_text)
                # Markdown で整形表示: 判定、詳細、根拠一覧（各抜粋は expander で展開）
                res_symbol = b_result.get("result") or b_result.get("status") or "△"
                st.markdown(f"**判定: {res_symbol}**")

                # 詳細説明があれば表示
                details = b_result.get("details") or b_result.get("detail") or b_result.get("notes")
                if details:
                    st.markdown("**説明:**")
                    st.text(details if isinstance(details, str) else json.dumps(details, ensure_ascii=False, indent=2))

                # 根拠を表示
                evs = b_result.get("evidence_normalized") or []
                if evs:
                    st.markdown("**根拠 (参照文書と抜粋):**")
                    for i, e in enumerate(evs, 1):
                        src = e.get("source") or "(unknown)"
                        excerpt = e.get("excerpt") or ""
                        with st.expander(f"{i}. {src}"):
                            ex = excerpt.replace("\r\n", "\n").replace("\r", "\n").strip()
                            st.text(ex)
                else:
                    st.info("(根拠情報はありません)")

                with st.expander("（参考）整形済みテキスト（生）"):
                    st.text(b_text_clean)

            except Exception as e:
                st.error("評価に失敗しました。ログを確認してください。")
                st.exception(e)

elif page == "RAG 質問":
    st.header("RAG 質問 (システム情報に関する QA)")
    q = st.text_input("質問を入力してください")
    if st.button("質問実行"):
        if not q:
            st.warning("質問を入力してください")
        else:
            try:
                docs = backend.retrieve_related_docs(vectordb, q, k=topk)
                context = "\n\n".join([f"[src:{d.metadata.get('source')}]\n{d.page_content}" for d in docs])
                system = "あなたはシステム情報の検索アシスタントです。ユーザの質問に、関連するドキュメントを参照して簡潔に答えてください。\n\n重要: 回答は必ず日本語で行ってください。"
                messages = [backend.SystemMessage(content=system), backend.HumanMessage(content=f"質問: {q}\n\n参照文書:\n{context}" )]
                resp = llm(messages)
                # 出力は小さく表示
                st.markdown("**回答:**")
                st.text(resp.content)
            except Exception as e:
                st.error("QA 実行でエラーが発生しました")
                st.exception(e)

st.caption("この UI は main2.py の関数を再利用しています。main2.py を大きく変更せずにフロントエンドを提供します。")
