"""Phần dùng chung của hai lớp bằng chứng: `critic` và `citation_checker`.

Cả hai cùng hỏi MỘT câu: "câu trích này có nằm nguyên văn trong MỘT DÒNG
của một tài liệu mà lượt chạy đã truy xuất không?" — và phải hỏi ĐÚNG như
`arena/scorer.py` hỏi (`_norm`, `_norm_lines`, `_supports`, `retrieved`).
Hỏi lệch đi một chút là layer giữ lại claim mà scorer chấm `HALLUCINATED`,
hoặc xoá mất claim mà scorer chấm `SUPPORTED`.

VÌ SAO KHÔNG CHỈ DÙNG `text in ctx.observed_text`:
  * scorer so khớp SAU KHI chuẩn hoá (NFC, casefold, gộp khoảng trắng).
    Mô hình thật hay xuống dòng/đổi hoa thường khác tài liệu một chút;
    so khớp thô sẽ xoá nhầm một claim mà scorer vẫn chấm đúng.
  * scorer chỉ nhận trích dẫn gọn trong MỘT DÒNG; `observed_text` là cả
    khối nối lại, một câu vắt qua hai dòng vẫn lọt.
  * scorer chỉ nhận tài liệu ĐÃ TRUY XUẤT (fetch, hoặc nằm trong kết quả
    search). Ghi lại đúng tập đó ở `wrap_tool_call` là cách duy nhất để
    gắn lại nguồn mà không bị `UNRETRIEVED`.

Module này KHÔNG BAO GIỜ sửa chữ của claim. Nó chỉ trả lời "có/không",
"tài liệu nào", và "đoạn con (substring) nào của chính câu mô hình viết là
trích dẫn được" — cắt bớt là hợp lệ, viết lại thì không.
"""

from __future__ import annotations

import re
import unicodedata

#: = `arena.scorer.MIN_SUPPORT_CHARS`: ngắn hơn thì không "đỡ" được gì.
MIN_QUOTE_CHARS = 12

#: = `arena.scorer.MAX_CLAIM_CHARS`: dài hơn là `OVERLONG`.
MAX_QUOTE_CHARS = 500

#: Khoá trong `ctx.state`: doc_id đã truy xuất, theo thứ tự gặp.
STATE_KEY = "evidence_doc_ids"

_WS_RE = re.compile(r"\s+")
#: Mã tài liệu trong kết quả `search` (một mảng JSON `{"doc_id": ...}`).
#: Chỉ khớp đúng khoá JSON, không khớp một mã tình cờ nằm trong snippet.
_SEARCH_HIT_RE = re.compile(r'"doc_id":\s*"(doc-\d{4})"')
#: Đơn vị để cắt câu: một từ, hoặc một dấu câu đứng riêng.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def norm(text) -> str:
    """Đúng `arena.scorer._norm`: NFC, casefold, gộp khoảng trắng."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def note_retrieval(ctx, name, args, result, calls_before) -> None:
    """Ghi lại những tài liệu scorer sẽ coi là "đã truy xuất".

    Gọi từ `wrap_tool_call` SAU `call(...)`. Chỉ tính khi công cụ thật sự
    chạy (`ctx.tools.calls` tăng) — một lượt bị `budget_policy` chặn thì
    không có sự kiện `tool_call` nào trên trace, nên cũng không truy xuất
    được gì.
    """
    if ctx.corpus is None or ctx.tools.calls == calls_before:
        return
    seen = ctx.state.setdefault(STATE_KEY, {})
    args = args if isinstance(args, dict) else {}
    if name == "fetch_doc":
        doc_id = args.get("doc_id")
        if isinstance(doc_id, str) and ctx.corpus.get(doc_id) is not None:
            seen[doc_id] = True
    elif name == "search" and result is not None:
        for doc_id in _SEARCH_HIT_RE.findall(getattr(result, "content", "") or ""):
            if ctx.corpus.get(doc_id) is not None:
                seen[doc_id] = True


class Evidence:
    """Các dòng (đã chuẩn hoá) của mọi tài liệu lượt chạy đã truy xuất."""

    def __init__(self, ctx) -> None:
        self.lines: dict[str, tuple] = {}
        if ctx.corpus is None:
            return
        for doc_id in ctx.state.get(STATE_KEY, {}):
            doc = ctx.corpus.get(doc_id)
            if doc is not None:
                self.lines[doc_id] = tuple(
                    line for line in (norm(raw) for raw in doc.body.splitlines()) if line
                )
        # Mọi dòng nối bằng "\n": một chuỗi đã chuẩn hoá không chứa "\n",
        # nên "nằm trong blob" đúng bằng "nằm trong MỘT dòng nào đó".
        self._blob = "\n".join(line for lines in self.lines.values() for line in lines)

    def quotes(self, doc_id, text) -> bool:
        """`doc_id` đã truy xuất VÀ có một dòng chứa nguyên văn `text`."""
        needle = norm(text)
        if len(needle) < MIN_QUOTE_CHARS or not isinstance(doc_id, str):
            return False
        return any(needle in line for line in self.lines.get(doc_id, ()))

    def source_of(self, text, prefer=None):
        """doc_id đã truy xuất có một dòng chứa `text`; ưu tiên `prefer`."""
        if self.quotes(prefer, text):
            return prefer
        return next((doc_id for doc_id in self.lines if self.quotes(doc_id, text)), None)

    def _quotable(self, text: str) -> bool:
        return norm(text) in self._blob

    def pieces(self, text: str, min_chars: int = 30, min_share: float = 0.35) -> list:
        """Các đoạn con của `text` trích được nguyên văn, theo thứ tự.

        Lấy đoạn trích được DÀI NHẤT (theo ranh giới từ/dấu câu), rồi làm
        tiếp với phần bên trái và bên phải nó. Mỗi đoạn là substring của
        chính chữ mô hình viết, nên vẫn qua kiểm tra provenance. Đoạn quá
        ngắn (< `min_chars`, hoặc < `min_share` độ dài câu) bị bỏ: đó là
        mẩu chung chung ("Công ty CP Hậu cần ...") chứ không phải trích dẫn.

        Bắt được ba kiểu hỏng thường gặp: câu ghép nửa câu của HAI tài
        liệu (mỗi nửa là một đoạn), câu vắt qua hai dòng, và câu bị thêm
        dấu chấm / dấu nháy ở hai đầu.
        """
        floor = max(min_chars, min_share * len(norm(text)))
        found: list[tuple] = []

        def split(lo: int, hi: int) -> None:
            spans = [(m.start() + lo, m.end() + lo) for m in _TOKEN_RE.finditer(text[lo:hi])]
            best = None
            for i, (start, _) in enumerate(spans):
                k = i
                # Đơn điệu: đoạn dài hơn không trích được thì dừng.
                while k < len(spans) and self._quotable(text[start:spans[k][1]]):
                    k += 1
                if k > i and (best is None or spans[k - 1][1] - start > best[1] - best[0]):
                    best = (start, spans[k - 1][1])
            if best is None or len(norm(text[best[0]:best[1]])) < floor:
                return
            split(lo, best[0])
            found.append(best)
            split(best[1], hi)

        split(0, len(text))
        return [text[start:end] for start, end in found]


def clip(text: str, limit: int = MAX_QUOTE_CHARS - 20) -> str:
    """Cắt bớt đuôi cho khỏi `OVERLONG` (substring — vẫn là trích dẫn)."""
    if len(norm(text)) <= MAX_QUOTE_CHARS:
        return text
    cut = text.rfind(" ", 0, limit)
    return text[: cut if cut > 0 else limit].rstrip()
