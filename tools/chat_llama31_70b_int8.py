#!/usr/bin/env python3
"""Practical terminal chat for CPU-resident Llama 3.1 70B W8A8."""

import argparse
import atexit
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    Int8DoubleBufferRuntime,
    Int8ResidentDeviceArena,
    Llama31DecodeExecutor,
    SamplingConfig,
    VocabStreamingRuntime,
    build_llama31_70b_int8_plan,
    collect_stop_token_ids,
    create_int8_weight_store,
    first_stop_string,
    render_chat_prompt,
    select_next_token,
)


DEFAULT_CHECKPOINT = Path(
    "/ssd/cascade-llm/models/Llama-3.1-70B-Instruct-W8A8"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. Answer in the language used by "
    "the user unless they explicitly request another language. Be honest "
    "about uncertainty and do not claim access to tools or current data."
)
SETTING_TYPES = {
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "min_p": float,
    "repetition_penalty": float,
    "presence_penalty": float,
    "frequency_penalty": float,
    "repetition_window": int,
    "min_new_tokens": int,
    "max_new_tokens": int,
    "max_context_tokens": int,
    "seed": int,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Keep Llama 3.1 70B W8A8 weights resident in CPU memory and "
            "serve a persistent single-GPU terminal chat."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="full_pinned",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-window", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--stop",
        action="append",
        default=[],
        help="Custom decoded-text stop string; may be repeated",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        help="Run one non-interactive turn and exit",
    )
    prompt_group.add_argument(
        "--prompt-file",
        type=Path,
        help="Read one non-interactive prompt from a UTF-8 file",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Append one JSON object per completed turn",
    )
    parser.add_argument(
        "--session",
        type=Path,
        help="Load this session JSON if present and autosave after changes",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path("~/.cache/cascade-llm/readline_history"),
        help="Persistent terminal input history",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Print each response only after generation finishes",
    )
    return parser.parse_args()


class ConsoleRenderer:
    """Stream decoded text while withholding possible custom stop suffixes."""

    def __init__(self, stop_strings):
        self.printed = ""
        stop_holdback = max(
            (len(item) - 1 for item in stop_strings if item),
            default=0,
        )
        self.holdback = max(4, stop_holdback)

    def update(self, text):
        if not text.startswith(self.printed):
            return
        safe_end = max(len(self.printed), len(text) - self.holdback)
        if safe_end > len(self.printed):
            print(text[len(self.printed) : safe_end], end="", flush=True)
            self.printed = text[:safe_end]

    def finish(self, text):
        if text.startswith(self.printed):
            print(text[len(self.printed) :], flush=True)
        elif not self.printed:
            print(text, flush=True)
        else:
            print("\n[final] {}".format(text), flush=True)
        self.printed = text


class ChatSession:
    SESSION_SCHEMA = 1

    def __init__(
        self,
        tokenizer,
        config,
        executor,
        runtime,
        device,
        sampling,
        max_context_tokens,
        stop_token_ids,
        stop_strings,
        generator,
        seed,
        system_prompt,
        transcript=None,
        session_path=None,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.executor = executor
        self.runtime = runtime
        self.device = device
        self.sampling = sampling.validate()
        self.max_context_tokens = int(max_context_tokens)
        self._validate_model_limits(
            self.sampling.top_k,
            self.max_context_tokens,
        )
        self.stop_token_ids = frozenset(int(item) for item in stop_token_ids)
        self.stop_strings = list(dict.fromkeys(stop_strings))
        self.generator = generator
        self.seed = int(seed)
        self.system_prompt = system_prompt
        self.transcript = transcript
        self.session_path = session_path
        self.messages = []
        self.last_stats = None
        self.reset(autosave=False)

    def _validate_model_limits(self, top_k, max_context_tokens):
        if top_k > int(self.config.vocab_size):
            raise ValueError(
                "top_k exceeds model vocabulary size {}".format(
                    self.config.vocab_size
                )
            )
        model_context = int(
            getattr(self.config, "max_position_embeddings", 0) or 0
        )
        if model_context and max_context_tokens > model_context:
            raise ValueError(
                "max_context_tokens exceeds model limit {}".format(
                    model_context
                )
            )

    def _base_messages(self):
        if not self.system_prompt:
            return []
        return [{"role": "system", "content": self.system_prompt}]

    def reset(self, autosave=True):
        self.executor.kv_cache.clear()
        self.messages = self._base_messages()
        self.last_stats = None
        if autosave:
            self.autosave()

    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        self.reset()

    def settings(self):
        result = self.sampling.as_dict()
        result.update(
            {
                "max_context_tokens": self.max_context_tokens,
                "seed": self.seed,
                "stop_strings": list(self.stop_strings),
            }
        )
        return result

    def set_setting(self, name, raw_value):
        name = name.strip().replace("-", "_")
        if name not in SETTING_TYPES:
            raise ValueError(
                "unknown setting {}; choose from {}".format(
                    name,
                    ", ".join(sorted(SETTING_TYPES)),
                )
            )
        value = SETTING_TYPES[name](raw_value)
        if name == "max_context_tokens":
            if value <= self.sampling.max_new_tokens:
                raise ValueError(
                    "max_context_tokens must exceed max_new_tokens"
                )
            self._validate_model_limits(self.sampling.top_k, value)
            self.max_context_tokens = value
        elif name == "seed":
            self.seed = value
            self.generator.manual_seed(value)
        else:
            old_value = getattr(self.sampling, name)
            setattr(self.sampling, name, value)
            try:
                self.sampling.validate()
                if self.max_context_tokens <= self.sampling.max_new_tokens:
                    raise ValueError(
                        "max_new_tokens must be below max_context_tokens"
                    )
                self._validate_model_limits(
                    self.sampling.top_k,
                    self.max_context_tokens,
                )
            except Exception:
                setattr(self.sampling, name, old_value)
                raise
            if name == "top_k":
                self.executor.top_k = value
        self.autosave()
        return value

    def add_stop_string(self, value):
        if not value:
            raise ValueError("stop string must not be empty")
        if value not in self.stop_strings:
            self.stop_strings.append(value)
        self.autosave()

    def clear_stop_strings(self):
        self.stop_strings = []
        self.autosave()

    def undo(self):
        system_offset = (
            1
            if self.messages and self.messages[0]["role"] == "system"
            else 0
        )
        if len(self.messages) - system_offset < 2:
            raise ValueError("there is no completed turn to undo")
        if self.messages[-2]["role"] != "user" or self.messages[-1]["role"] != "assistant":
            raise ValueError("conversation does not end in a completed turn")
        removed = self.messages[-2:]
        del self.messages[-2:]
        self.executor.kv_cache.clear()
        self.last_stats = None
        self.autosave()
        return removed

    def prepare_retry(self):
        removed = self.undo()
        return removed[0]["content"]

    def _model_step(self, input_ids, token_history, banned_token_ids):
        started = time.perf_counter()
        state = self.executor.begin(input_ids)
        state = self.runtime.run(self.executor, state)
        state = self.executor.finish(state)
        next_token = select_next_token(
            state.topk_values,
            state.topk_indices,
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            min_p=self.sampling.min_p,
            repetition_penalty=self.sampling.repetition_penalty,
            presence_penalty=self.sampling.presence_penalty,
            frequency_penalty=self.sampling.frequency_penalty,
            token_history=token_history,
            repetition_window=self.sampling.repetition_window,
            banned_token_ids=banned_token_ids,
            generator=self.generator,
        )
        torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return next_token, elapsed_ms

    def ask(self, user_text, stream=True):
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("user message must not be empty")
        previous_messages = [dict(item) for item in self.messages]
        candidate = previous_messages + [{"role": "user", "content": user_text}]
        rendered = render_chat_prompt(
            self.tokenizer,
            candidate,
            max_context_tokens=self.max_context_tokens,
            max_new_tokens=self.sampling.max_new_tokens,
        )
        self.messages = list(rendered.messages)

        # Multi-token append with a past cache is not supported by the current
        # executor. Re-prefill retained history while keeping all CPU weights
        # and GPU streaming buffers resident.
        self.executor.kv_cache.clear()
        current = rendered.input_ids.to(self.device)
        generated = []
        token_history = rendered.input_ids.reshape(-1).tolist()
        token_latencies_ms = []
        decoded = ""
        stop_reason = "length"
        matched_stop = None
        renderer = ConsoleRenderer(self.stop_strings) if stream else None
        turn_started = time.perf_counter()

        try:
            with torch.inference_mode():
                for _ in range(self.sampling.max_new_tokens):
                    banned = (
                        self.stop_token_ids
                        if len(generated) < self.sampling.min_new_tokens
                        else ()
                    )
                    next_token, elapsed_ms = self._model_step(
                        current,
                        token_history,
                        banned,
                    )
                    token_id = int(next_token.item())
                    token_latencies_ms.append(elapsed_ms)
                    if token_id in self.stop_token_ids:
                        stop_reason = "eos"
                        break
                    generated.append(token_id)
                    token_history.append(token_id)
                    decoded = self.tokenizer.decode(
                        generated,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    if len(generated) >= self.sampling.min_new_tokens:
                        match = first_stop_string(
                            decoded,
                            self.stop_strings,
                        )
                        if match is not None:
                            position, matched_stop = match
                            decoded = decoded[:position]
                            stop_reason = "custom"
                            break
                    if renderer is not None:
                        renderer.update(decoded)
                    current = next_token
        except BaseException:
            self.messages = previous_messages
            self.executor.kv_cache.clear()
            if renderer is not None:
                renderer.finish(decoded)
            raise

        if renderer is not None:
            renderer.finish(decoded)
        wall_seconds = time.perf_counter() - turn_started
        self.messages.append({"role": "assistant", "content": decoded})
        self.last_stats = {
            "prompt_tokens": int(rendered.input_ids.numel()),
            "generated_tokens": len(generated),
            "model_steps": len(token_latencies_ms),
            "dropped_messages": rendered.dropped_messages,
            "stop_reason": stop_reason,
            "matched_stop": matched_stop,
            "wall_seconds": wall_seconds,
            "tokens_per_second": (
                len(generated) / wall_seconds if wall_seconds else 0.0
            ),
            "model_steps_per_second": (
                len(token_latencies_ms) / wall_seconds
                if wall_seconds
                else 0.0
            ),
            "token_latency_ms": token_latencies_ms,
        }
        self._append_transcript(user_text, decoded)
        self.autosave()
        return decoded, dict(self.last_stats)

    def _append_transcript(self, user_text, assistant_text):
        if self.transcript is None:
            return
        self.transcript.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "user": user_text,
            "assistant": assistant_text,
            "settings": self.settings(),
            "stats": self.last_stats,
        }
        with self.transcript.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")

    def session_payload(self):
        return {
            "schema_version": self.SESSION_SCHEMA,
            "model": getattr(self.config, "_name_or_path", None),
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "system_prompt": self.system_prompt,
            "messages": self.messages,
            "settings": self.settings(),
        }

    def save_session(self, path=None):
        path = Path(path or self.session_path) if (path or self.session_path) else None
        if path is None:
            raise ValueError("no session path was configured")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(
                self.session_payload(),
                destination,
                ensure_ascii=False,
                indent=2,
            )
            destination.write("\n")
        temporary.replace(path)
        self.session_path = path
        return path

    def autosave(self):
        if self.session_path is not None:
            self.save_session(self.session_path)

    def load_session(self, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if payload.get("schema_version") != self.SESSION_SCHEMA:
            raise ValueError("unsupported session schema")
        messages = payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError("session messages must be a list")
        for message in messages:
            if (
                not isinstance(message, dict)
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise ValueError("session contains an invalid message")

        settings = payload.get("settings") or {}
        loaded_sampling = SamplingConfig(
            **{
                key: settings.get(key, value)
                for key, value in self.sampling.as_dict().items()
            }
        ).validate()
        max_context = int(
            settings.get("max_context_tokens", self.max_context_tokens)
        )
        if max_context <= loaded_sampling.max_new_tokens:
            raise ValueError("saved context is smaller than output budget")
        self._validate_model_limits(loaded_sampling.top_k, max_context)
        self.system_prompt = payload.get("system_prompt") or ""
        self.messages = [dict(item) for item in messages]
        self.sampling = loaded_sampling
        self.max_context_tokens = max_context
        self.seed = int(settings.get("seed", self.seed))
        self.generator.manual_seed(self.seed)
        loaded_stops = settings.get("stop_strings", [])
        if (
            not isinstance(loaded_stops, list)
            or not all(isinstance(item, str) for item in loaded_stops)
        ):
            raise ValueError("saved stop_strings must be a list of strings")
        self.stop_strings = list(dict.fromkeys(loaded_stops))
        self.executor.top_k = self.sampling.top_k
        self.executor.kv_cache.clear()
        self.last_stats = None
        self.session_path = path
        return path


def sampling_from_args(args):
    return SamplingConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_window=args.repetition_window,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
    ).validate()


def validate_args(args):
    try:
        sampling_from_args(args)
    except ValueError as error:
        raise SystemExit(str(error))
    if args.max_context_tokens <= args.max_new_tokens:
        raise SystemExit(
            "--max-context-tokens must be larger than --max-new-tokens"
        )
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    if not args.checkpoint.is_dir():
        raise SystemExit(
            "checkpoint directory does not exist: {}".format(args.checkpoint)
        )
    if args.prompt_file is not None and not args.prompt_file.is_file():
        raise SystemExit(
            "prompt file does not exist: {}".format(args.prompt_file)
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")


def initialize(args):
    validate_args(args)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    print("[1/4] Loading tokenizer and configuration...", flush=True)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        local_files_only=True,
    )
    if not tokenizer.chat_template:
        raise SystemExit("checkpoint tokenizer has no chat template")

    print(
        "[2/4] Allocating the {} CPU weight store...".format(
            args.weight_store
        ),
        flush=True,
    )
    plan = build_llama31_70b_int8_plan(args.granularity)
    allocation_started = time.perf_counter()
    store = create_int8_weight_store(
        plan,
        args.weight_store,
        slot_count=args.slots,
    )
    allocation_seconds = time.perf_counter() - allocation_started

    print(
        "[3/4] Reading {:.2f} GB of checkpoint tensors from SSD...".format(
            plan.host_arena_bytes / 1e9
        ),
        flush=True,
    )
    load_started = time.perf_counter()
    store.load_checkpoint(args.checkpoint)
    load_seconds = time.perf_counter() - load_started

    print("[4/4] Initializing GPU slots and executor...", flush=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident = Int8ResidentDeviceArena(plan, store, device)
    runtime = Int8DoubleBufferRuntime(
        plan,
        store,
        resident,
        device=device,
        slot_count=args.slots,
        profile=False,
    )
    vocab_runtime = VocabStreamingRuntime(
        plan,
        store,
        runtime,
        profile=False,
    )
    sampling = sampling_from_args(args)
    executor = Llama31DecodeExecutor(
        config,
        resident,
        vocab_runtime=vocab_runtime,
        top_k=sampling.top_k,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    session = ChatSession(
        tokenizer=tokenizer,
        config=config,
        executor=executor,
        runtime=runtime,
        device=device,
        sampling=sampling,
        max_context_tokens=args.max_context_tokens,
        stop_token_ids=collect_stop_token_ids(config, tokenizer),
        stop_strings=args.stop,
        generator=generator,
        seed=args.seed,
        system_prompt=args.system_prompt,
        transcript=args.transcript,
        session_path=args.session,
    )
    if args.session is not None and args.session.is_file():
        session.load_session(args.session)
        print("Loaded session: {}".format(args.session), flush=True)
    startup = {
        "allocation_seconds": allocation_seconds,
        "load_seconds": load_seconds,
        "pinned_cpu_gib": (
            store.pinned_cpu_bytes + vocab_runtime.extra_pinned_cpu_bytes
        )
        / 1024**3,
        "planned_gpu_weight_gib": runtime.stats["weight_gpu_bytes"]
        / 1024**3,
        "model_id": plan.model_id,
        "weight_store": args.weight_store,
        "granularity": args.granularity,
        "slots": args.slots,
        "device": str(device),
    }
    return session, store, startup


def print_ready(session, startup):
    print()
    print(
        "Ready: allocation={:.1f}s, SSD->CPU={:.1f}s, "
        "pinned CPU={:.2f} GiB, planned GPU weight path={:.2f} GiB".format(
            startup["allocation_seconds"],
            startup["load_seconds"],
            startup["pinned_cpu_gib"],
            startup["planned_gpu_weight_gib"],
        )
    )
    print_config(session)


def print_config(session):
    settings = session.settings()
    print(
        "Generation: max={max_new_tokens}, min={min_new_tokens}, "
        "context={max_context_tokens}, temperature={temperature}, "
        "top_k={top_k}, top_p={top_p}, min_p={min_p}, "
        "repeat={repetition_penalty}".format(**settings)
    )
    print(
        "Penalties: presence={presence_penalty}, frequency={frequency_penalty}, "
        "window={repetition_window}, seed={seed}".format(**settings)
    )
    print("Custom stops: {}".format(session.stop_strings or "none"))


def print_stats(stats):
    if stats is None:
        print("No completed turn yet.")
        return
    print(
        "prompt={prompt_tokens}, output={generated_tokens}, "
        "steps={model_steps}, wall={wall_seconds:.2f}s, "
        "visible_speed={tokens_per_second:.3f} token/s, "
        "stop={stop_reason}, dropped_messages={dropped_messages}".format(
            **stats
        )
    )


def print_history(session):
    if not session.messages:
        print("History is empty.")
        return
    for index, message in enumerate(session.messages):
        content = message["content"].replace("\n", "\\n")
        if len(content) > 240:
            content = content[:237] + "..."
        print("{:02d} {:9s} {}".format(index, message["role"], content))


def read_multiline():
    print("Enter multiple lines; a line containing only '.' finishes.")
    lines = []
    while True:
        try:
            line = input("... ")
        except EOFError:
            break
        if line == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def print_help():
    print(
        """
/help                  Show this help
/reset                 Clear conversation history
/history               Show retained messages
/undo                  Remove the latest user/assistant turn
/retry                 Regenerate the latest answer
/multi                 Enter a multiline user message; '.' finishes
/stats                 Show last-turn timing
/config                Show generation settings
/set NAME VALUE         Change a setting at runtime
/stop list              List custom stop strings
/stop add TEXT          Add a custom stop string
/stop clear             Remove all custom stop strings
/system TEXT            Replace system prompt and reset history
/save [PATH]            Save session JSON
/load PATH              Load session JSON
/exit                   Exit and release CPU/GPU memory
""".strip()
    )


def handle_command(session, text):
    command, _, argument = text.partition(" ")
    argument = argument.strip()
    if command in {"/exit", "/quit"}:
        return "exit", None
    if command == "/help":
        print_help()
    elif command == "/reset":
        session.reset()
        print("Conversation history cleared.")
    elif command == "/history":
        print_history(session)
    elif command == "/undo":
        session.undo()
        print("Latest turn removed.")
    elif command == "/retry":
        return "ask", session.prepare_retry()
    elif command == "/multi":
        return "ask", read_multiline()
    elif command == "/stats":
        print_stats(session.last_stats)
    elif command == "/config":
        print_config(session)
    elif command == "/set":
        name, separator, value = argument.partition(" ")
        if not separator:
            raise ValueError("usage: /set NAME VALUE")
        converted = session.set_setting(name, value.strip())
        print("{}={}".format(name.replace("-", "_"), converted))
    elif command == "/stop":
        action, _, value = argument.partition(" ")
        if action in {"", "list"}:
            print("Custom stops: {}".format(session.stop_strings or "none"))
        elif action == "add":
            session.add_stop_string(value)
            print("Custom stop added.")
        elif action == "clear":
            session.clear_stop_strings()
            print("Custom stops cleared.")
        else:
            raise ValueError("usage: /stop list|add TEXT|clear")
    elif command == "/system":
        if not argument:
            raise ValueError("usage: /system TEXT")
        session.set_system_prompt(argument)
        print("System prompt updated; conversation history cleared.")
    elif command == "/save":
        path = session.save_session(argument or None)
        print("Session saved: {}".format(path))
    elif command == "/load":
        if not argument:
            raise ValueError("usage: /load PATH")
        path = session.load_session(argument)
        print("Session loaded: {}".format(path))
    else:
        raise ValueError("unknown command {}; use /help".format(command))
    return "continue", None


def repl(session, stream=True):
    print("Type /help for commands.", flush=True)
    pending = None
    while True:
        if pending is None:
            try:
                user_text = input("\nuser> ").strip()
            except EOFError:
                print("\nExiting.")
                return
            except KeyboardInterrupt:
                print("\nUse /exit to quit.")
                continue
        else:
            user_text = pending
            pending = None
        if not user_text:
            continue
        if user_text.startswith("/"):
            try:
                action, value = handle_command(session, user_text)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print("Error: {}".format(error))
                continue
            if action == "exit":
                return
            if action == "ask":
                if not value:
                    print("No message was entered.")
                    continue
                user_text = value
            else:
                continue

        print("assistant> ", end="", flush=True)
        try:
            text, stats = session.ask(user_text, stream=stream)
        except KeyboardInterrupt:
            print("\nGeneration interrupted; conversation was rolled back.")
            continue
        except (RuntimeError, ValueError) as error:
            print("\nGeneration failed: {}".format(error))
            continue
        if not stream:
            print(text)
        print_stats(stats)


def configure_readline(path):
    try:
        import readline
    except ImportError:
        return
    path = path.expanduser()
    if path.is_file():
        try:
            readline.read_history_file(str(path))
        except OSError:
            pass

    def save_history():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            readline.write_history_file(str(path))
        except OSError:
            pass

    atexit.register(save_history)


def main():
    args = parse_args()
    configure_readline(args.history_file)
    store = None
    try:
        session, store, startup = initialize(args)
        print_ready(session, startup)
        prompt = args.prompt
        if args.prompt_file is not None:
            prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        if prompt is not None:
            print("assistant> ", end="", flush=True)
            text, stats = session.ask(prompt, stream=not args.no_stream)
            if args.no_stream:
                print(text)
            print_stats(stats)
            return
        repl(session, stream=not args.no_stream)
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    main()
