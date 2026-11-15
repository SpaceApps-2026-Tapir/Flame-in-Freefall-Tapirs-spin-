"""IEEE-style presentation of validated retrieval citations."""
import re


def ieee_citations(text, passages):
    """Number documents by first citation; keep page details in source excerpts."""
    order = []
    for match in re.finditer(r"\[S(\d+)\]", text or ""):
        index = int(match[1]) - 1
        if 0 <= index < len(passages) and index not in order:
            order.append(index)
    cited = set(order)
    order.extend(i for i in range(len(passages)) if i not in cited)
    numbers, references, labels = {}, [], {}
    for index in order:
        passage = passages[index]
        url = passage.get("source_page_url") or passage.get("pdf_url", "")
        key = passage.get("document_id") or url or passage["title"]
        if key not in numbers:
            numbers[key] = len(numbers) + 1
            # Metadata currently contains titles and URLs, not verified authors or dates.
            references.append({"number": numbers[key], "title": passage["title"], "url": url,
                               "cited": index in cited})
        labels[index] = numbers[key]

    def replace(match):
        index = int(match[1]) - 1
        if index not in labels:
            return match[0]
        return f"[{labels[index]}, p. {passages[index]['page']}]"

    return re.sub(r"\[S(\d+)\]", replace, text or ""), references, labels
