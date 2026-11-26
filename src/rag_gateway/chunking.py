import re


def chunk_policy(chunk_size: int, overlap: int) -> str:
    return f"heading-v2:{chunk_size}:{overlap}"


def _split_text(normalized: str, chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        while start < len(normalized) and normalized[start].isspace():
            start += 1
        if start == len(normalized):
            break
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + 1, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end].strip())
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    headings: list[tuple[int, str]] = []
    body: list[str] = []
    pending_heading = False
    minimum_body = max(1, chunk_size // 4)

    def flush() -> None:
        nonlocal pending_heading
        normalized = " ".join(" ".join(body).split())
        body.clear()
        if not normalized and not pending_heading:
            return
        if not normalized:
            heading_text = " ".join(heading for _, heading in headings)
            chunks.extend(_split_text(heading_text, chunk_size, min(overlap, chunk_size // 2)))
        else:
            prefix_parts: list[str] = []
            for _, heading in reversed(headings):
                candidate = " ".join([heading, *prefix_parts])
                if len(candidate) + 1 + minimum_body > chunk_size:
                    break
                prefix_parts.insert(0, heading)
            if headings and not prefix_parts:
                chunks.extend(_split_text(headings[-1][1], chunk_size, min(overlap, chunk_size // 2)))
            prefix = " ".join(prefix_parts)
            budget = chunk_size - len(prefix) - (1 if prefix else 0)
            body_overlap = min(overlap, budget // 2) if prefix else overlap
            for part in _split_text(normalized, budget, body_overlap):
                chunks.append(f"{prefix} {part}" if prefix else part)
        pending_heading = False

    lines = text.splitlines()
    fence: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence_match:
            marker, suffix = fence_match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = None
            body.append(line)
            index += 1
            continue

        match = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$", line) if fence is None else None
        level = len(match.group(1)) if match else 0
        title = match.group(2).strip() if match else ""
        if not match and fence is None and line.strip() and index + 1 < len(lines):
            underline = re.fullmatch(r" {0,3}(=+|-+)[ \t]*", lines[index + 1])
            if underline:
                level = 1 if underline.group(1).startswith("=") else 2
                title = line.strip()
                index += 1

        if level and title:
            next_heading = f"{'#' * level} {title}"
            if (
                any(part.strip() for part in body)
                or (headings and level <= headings[-1][0])
                or len(" ".join([*(heading for _, heading in headings), next_heading]))
                + 1 + minimum_body > chunk_size
            ):
                flush()
            headings[:] = [(depth, heading) for depth, heading in headings if depth < level]
            headings.append((level, next_heading))
            pending_heading = True
        else:
            body.append(line)
        index += 1

    flush()
    return chunks
