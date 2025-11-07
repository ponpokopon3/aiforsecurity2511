import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage
import json
import certifi

# certifi の CA を明示的に SSL 証明書ファイルとして使う（httpx/openai の検証用）
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# フォールバック対応で ConversationBufferMemory をインポート（複数バージョン対応）
ConversationBufferMemory = None
try:
    from langchain.memory import ConversationBufferMemory  # 標準的な場所
    ConversationBufferMemory = ConversationBufferMemory
except Exception:
    try:
        from langchain_community.memory import ConversationBufferMemory  # community 配下
        ConversationBufferMemory = ConversationBufferMemory
    except Exception:
        # 見つからない場合は互換ラッパーを定義
        class ConversationBufferMemory:
            def __init__(self, return_messages: bool = True, memory_key: str = "chat_history"):
                self.return_messages = return_messages
                self.memory_key = memory_key
                # chat_memory をエミュレート（load/save で使われる形に）
                class ChatMemory:
                    def __init__(self):
                        self.messages = []
                    def add_user_message(self, msg: str):
                        self.messages.append(HumanMessage(content=msg))
                    def add_ai_message(self, msg: str):
                        self.messages.append(AIMessage(content=msg))
                self.chat_memory = ChatMemory()
            def save_context(self, inputs: dict, outputs: dict):
                if "input" in inputs:
                    self.chat_memory.add_user_message(inputs["input"])
                if "output" in outputs:
                    self.chat_memory.add_ai_message(outputs["output"])
            def clear(self):
                self.chat_memory.messages = []

class RAGChatBot:
    def __init__(self, api_key: str, persist_directory: str = "./chroma_db"):
        """RAGチャットボットの初期化"""
        self.api_key = api_key
        self.persist_directory = persist_directory
        
        # LLMとEmbeddingsの初期化
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model="gpt-4o",
            temperature=0.7,
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key
        )
        
        # LangChainのメモリを使用
        self.memory = ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
        
        # ベクターストアの初期化
        self.vectorstore = None
        self.retriever = None
        
        # RAGチェーンの初期化
        self.rag_chain = None
        
        self._setup_rag_chain()
    
    def load_documents(self, document_paths: List[str]):
        """文書をロードして分割、ベクターストアに追加"""
        documents = []
        
        for path in document_paths:
            path_obj = Path(path)
            
            if path_obj.is_file():
                if path_obj.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(path_obj))
                elif path_obj.suffix.lower() in ['.txt', '.md']:
                    loader = TextLoader(str(path_obj), encoding='utf-8')
                else:
                    print(f"サポートされていないファイル形式: {path}")
                    continue
                    
                documents.extend(loader.load())
                
            elif path_obj.is_dir():
                # ディレクトリの場合、サポートされるファイルを再帰的にロード
                loader = DirectoryLoader(
                    str(path_obj),
                    glob="**/*.{pdf,txt,md}",
                    loader_cls=TextLoader,
                    loader_kwargs={'encoding': 'utf-8'}
                )
                documents.extend(loader.load())
        
        if not documents:
            print("読み込める文書が見つかりませんでした。")
            return
        
        # テキスト分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # ベクターストアに追加
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(splits)
        
        # リトリーバーの更新
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
        
        # RAGチェーンの再構築
        self._setup_rag_chain()
        
        print(f"文書を正常に読み込みました。チャンク数: {len(splits)}")
    
    def _format_docs(self, docs):
        """文書をフォーマット"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _setup_rag_chain(self):
        """RAGチェーンのセットアップ"""
        # プロンプトテンプレート
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたは親切で知識豊富なAIアシスタントです。
以下のコンテキスト情報を参考にして、ユーザーの質問に日本語で回答してください。

コンテキスト情報:
{context}

回答する際の注意点:
- コンテキストに基づいて正確な情報を提供してください
- コンテキストに関連情報がない場合は、一般的な知識で回答し、その旨を明記してください
- 簡潔で分かりやすい回答を心がけてください"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        if self.retriever is not None:
            # RAGチェーン（文書検索あり）
            self.rag_chain = (
                RunnableParallel({
                    "context": self.retriever | self._format_docs,
                    "question": RunnablePassthrough(),
                    "chat_history": RunnableLambda(lambda _: self.memory.chat_memory.messages)
                })
                | prompt
                | self.llm
                | StrOutputParser()
            )
        else:
            # 通常のチャット（文書検索なし）
            simple_prompt = ChatPromptTemplate.from_messages([
                ("system", "あなたは親切で知識豊富なAIアシスタントです。日本語で回答してください。"),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            self.rag_chain = (
                RunnableParallel({
                    "question": RunnablePassthrough(),
                    "chat_history": RunnableLambda(lambda _: self.memory.chat_memory.messages)
                })
                | simple_prompt
                | self.llm
                | StrOutputParser()
            )
    
    def chat(self, question: str) -> str:
        """チャット実行"""
        if self.rag_chain is None:
            return "チャットシステムが初期化されていません。"
        
        try:
            # RAGチェーン実行
            response = self.rag_chain.invoke(question)
            
            # LangChainメモリに保存
            self.memory.save_context(
                {"input": question},
                {"output": response}
            )
            
            return response
        except Exception as e:
            return f"エラーが発生しました: {str(e)}"
    
    def clear_memory(self):
        """会話履歴をクリア"""
        self.memory.clear()
        print("会話履歴をクリアしました。")
    
    def save_memory(self, filepath: str):
        """会話履歴を保存"""
        try:
            messages = []
            for msg in self.memory.chat_memory.messages:
                if isinstance(msg, HumanMessage):
                    messages.append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"type": "ai", "content": msg.content})
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
            print(f"会話履歴を保存しました: {filepath}")
        except Exception as e:
            print(f"保存エラー: {str(e)}")
    
    def load_memory(self, filepath: str):
        """会話履歴を読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                messages = json.load(f)
            
            self.memory.clear()
            for msg in messages:
                if msg["type"] == "human":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg["type"] == "ai":
                    self.memory.chat_memory.add_ai_message(msg["content"])
            
            print(f"会話履歴を読み込みました: {filepath}")
        except Exception as e:
            print(f"読み込みエラー: {str(e)}")

def main():
    """メイン関数 - CUIインターフェース"""
    print("=== RAGチャットボット ===")
    
    # APIキー入力
    api_key = input("OpenAI APIキーを入力してください: ").strip()
    if not api_key:
        print("APIキーが必要です。")
        return
    
    # RAGチャットボット初期化
    bot = RAGChatBot(api_key)
    
    print("\n=== 初期設定 ===")
    print("文書を読み込みますか？ (y/n)")
    if input().lower() == 'y':
        print("文書のパス（ファイルまたはディレクトリ）をカンマ区切りで入力してください:")
        paths_input = input().strip()
        if paths_input:
            paths = [p.strip() for p in paths_input.split(',')]
            bot.load_documents(paths)
    
    print("\n=== チャット開始 ===")
    print("コマンド:")
    print("  /help - ヘルプ表示")
    print("  /clear - 会話履歴クリア")
    print("  /save <filepath> - 会話履歴保存")
    print("  /load <filepath> - 会話履歴読み込み")
    print("  /docs <paths> - 文書追加読み込み")
    print("  /quit - 終了")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # コマンド処理
            if user_input.startswith('/'):
                parts = user_input.split(' ', 1)
                command = parts[0].lower()
                
                if command == '/quit':
                    print("チャットを終了します。")
                    break
                elif command == '/help':
                    print("コマンド:")
                    print("  /help - ヘルプ表示")
                    print("  /clear - 会話履歴クリア")
                    print("  /save <filepath> - 会話履歴保存")
                    print("  /load <filepath> - 会話履歴読み込み")
                    print("  /docs <paths> - 文書追加読み込み")
                    print("  /quit - 終了")
                elif command == '/clear':
                    bot.clear_memory()
                elif command == '/save':
                    if len(parts) > 1:
                        bot.save_memory(parts[1])
                    else:
                        print("保存先のファイルパスを指定してください。")
                elif command == '/load':
                    if len(parts) > 1:
                        bot.load_memory(parts[1])
                    else:
                        print("読み込み元のファイルパスを指定してください。")
                elif command == '/docs':
                    if len(parts) > 1:
                        paths = [p.strip() for p in parts[1].split(',')]
                        bot.load_documents(paths)
                    else:
                        print("文書のパスを指定してください。")
                else:
                    print("不明なコマンドです。/help でヘルプを表示してください。")
                continue
            
            # チャット実行
            print("AI: ", end="", flush=True)
            response = bot.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\nチャットを終了します。")
            break
        except Exception as e:
            print(f"エラー: {str(e)}")

if __name__ == "__main__":
    main()
