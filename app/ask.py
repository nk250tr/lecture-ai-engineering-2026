"""第1回: 履歴を保った会話とモデル・質問ごとの計測を行う。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask --compare --model gemini-3.5-flash "LoRAとは？" "QLoRAとは？"

通常モードは会話履歴を保ち、回答後に続けて質問できる。
比較モードは各質問を独立したリクエストとしてモデルごとに実行し、
応答時間・トークン数・回答を表示する。
"""

import argparse
import time
from dataclasses import dataclass
from typing import Protocol

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


class ChatSession(Protocol):
    def send_message(self, message: str) -> types.GenerateContentResponse: ...


@dataclass
class Measurement:
    model: str
    question: str
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    answer: str | None
    error: str | None = None


def measure(question: str, model: str, send) -> Measurement:
    """API 呼び出し1回の応答時間・トークン数・回答を記録する。"""
    start = time.perf_counter()
    try:
        response = send()
    except errors.APIError as e:
        return Measurement(
            model=model,
            question=question,
            elapsed_seconds=time.perf_counter() - start,
            input_tokens=None,
            output_tokens=None,
            answer=None,
            error=api_error_message(e.code, e.message),
        )

    elapsed = time.perf_counter() - start
    usage = response.usage_metadata
    return Measurement(
        model=model,
        question=question,
        elapsed_seconds=elapsed,
        input_tokens=usage.prompt_token_count if usage else None,
        output_tokens=usage.candidates_token_count if usage else None,
        answer=response.text,
    )


def print_measurement(result: Measurement) -> None:
    print(f"[model] {result.model}")
    print(f"[応答時間] {result.elapsed_seconds:.2f} 秒")
    if result.input_tokens is None or result.output_tokens is None:
        print("[トークン数] 利用情報なし")
    else:
        print(f"[トークン数] 入力: {result.input_tokens} / 出力: {result.output_tokens}")
    if result.error:
        print(f"[エラー] {result.error}")
    else:
        print(result.answer or "")


def compare(client, questions: list[str], models: list[str]) -> None:
    """質問とモデルの全組み合わせを独立して実行し、測定値と回答を表示する。"""
    results = []
    for question in questions:
        for model in models:
            print(f"\n[質問] {question}")
            result = measure(
                question,
                model,
                lambda question=question, model=model: client.models.generate_content(
                    model=model, contents=question
                ),
            )
            results.append(result)
            print_measurement(result)

    print("\n--- 比較結果 ---")
    print("モデル | 質問 | 応答時間 | 入力token | 出力token | 結果")
    for result in results:
        input_tokens = result.input_tokens if result.input_tokens is not None else "-"
        output_tokens = result.output_tokens if result.output_tokens is not None else "-"
        status = "失敗" if result.error else "成功"
        print(
            f"{result.model} | {result.question} | {result.elapsed_seconds:.2f}秒 | "
            f"{input_tokens} | {output_tokens} | {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="会話履歴を保ちながら質問する／複数モデル・質問を計測して比較する"
    )
    parser.add_argument("questions", nargs="*", help="質問（比較時は複数指定できます）")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="質問とモデルの全組み合わせを独立して計測・比較する",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="比較するモデル（複数指定可。省略時は AIE_MODEL の設定を使う）",
    )
    args = parser.parse_args()

    if args.compare and not args.questions:
        parser.error("--compare には1つ以上の質問を指定してください")
    if args.models and not args.compare:
        parser.error("--model は --compare と一緒に指定してください")

    models = args.models or [MODEL]
    try:
        client = genai.Client(api_key=get_api_key())
        if args.compare:
            compare(client, args.questions, models)
            return

        chat = client.chats.create(model=MODEL)
        questions = args.questions or [DEFAULT_QUESTION]
        question_index = 0
        while True:
            question = (
                questions[question_index]
                if question_index < len(questions)
                else None
            )
            if question is None:
                try:
                    question = input(
                        "\n続けて質問してください（終了するには「終了」または exit）: "
                    )
                except (EOFError, KeyboardInterrupt):
                    print()
                    return
                if question.strip().lower() in {"終了", "exit", "quit"}:
                    return
                if not question.strip():
                    continue
            else:
                question_index += 1

            print(f"\n[質問] {question}")
            result = measure(question, MODEL, lambda: chat.send_message(question))
            print_measurement(result)
            if result.error:
                raise SystemExit(1)
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
