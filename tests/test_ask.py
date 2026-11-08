import sys
from types import SimpleNamespace

from app import ask as ask_module


class FakeChat:
    def __init__(self):
        self.questions = []

    def send_message(self, message):
        self.questions.append(message)
        return SimpleNamespace(text=f"回答: {message}", usage_metadata=None)


class FakeClient:
    def __init__(self, chat):
        self.chats = SimpleNamespace(create=lambda model: chat)


def test_main_keeps_one_chat_for_follow_up_questions(monkeypatch, capsys):
    chat = FakeChat()
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: FakeClient(chat))
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(ask_module, "MODEL", "test-model")
    monkeypatch.setattr(sys, "argv", ["app.ask", "最初の質問"])
    answers = iter(["", "続きの質問", "終了"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    ask_module.main()

    assert chat.questions == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert "回答: 最初の質問" in output
    assert "回答: 続きの質問" in output


def test_measure_records_elapsed_time_and_token_counts(monkeypatch):
    times = iter([10.0, 12.5])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(times))
    response = SimpleNamespace(
        text="回答",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=8),
    )

    result = ask_module.measure("質問", "test-model", lambda: response)

    assert result.elapsed_seconds == 2.5
    assert result.input_tokens == 12
    assert result.output_tokens == 8
    assert result.answer == "回答"


def test_compare_runs_each_question_for_each_model(capsys):
    calls = []

    class FakeModels:
        def generate_content(self, model, contents):
            calls.append((model, contents))
            return SimpleNamespace(
                text=f"{model} の回答",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=1, candidates_token_count=2
                ),
            )

    client = SimpleNamespace(models=FakeModels())

    ask_module.compare(client, ["質問1", "質問2"], ["model-a", "model-b"])

    assert calls == [
        ("model-a", "質問1"),
        ("model-b", "質問1"),
        ("model-a", "質問2"),
        ("model-b", "質問2"),
    ]
    output = capsys.readouterr().out
    assert "--- 比較結果 ---" in output
    assert "model-b | 質問2" in output
