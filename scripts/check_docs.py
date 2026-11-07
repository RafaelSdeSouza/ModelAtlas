"""Verify local links, fragment targets and exported gallery assets."""

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self, path: Path) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.errors: list[str] = []
        self.feed(path.read_text(encoding="utf-8"))

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("name"):
            self.ids.add(attrs["name"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.links.append(attrs[key])
        if (
            tag == "img"
            and "sphx_glr_plot_" in attrs.get("src", "")
            and "_thumb." not in attrs["src"]
            and not attrs.get("alt")
        ):
            self.errors.append(f"missing figure alternative text: {attrs.get('src')}")


def main(root: Path) -> None:
    root = root.resolve()
    pages = {
        path: Page(path)
        for path in root.rglob("*.html")
        if not path.relative_to(root).parts[0].startswith("_")
    }
    if root / "index.html" not in pages or root / "gallery" / "index.html" not in pages:
        raise SystemExit("documentation is missing an index or gallery")
    nlinks = 0
    errors = []
    for path, page in pages.items():
        errors.extend(f"{path.relative_to(root)}: {error}" for error in page.errors)
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            filename = unquote(url.path)
            target = ((path.parent / filename) if filename else path).resolve()
            if target.is_dir():
                target = target / "index.html"
            nlinks += 1
            if not target.is_file() or not target.is_relative_to(root):
                errors.append(f"{path.relative_to(root)}: missing local target {link!r}")
            elif (
                url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(root)}: missing fragment {link!r}")

    gallery_images = list((root / "_images").glob("sphx_glr_plot_*_[0-9][0-9][0-9].svg"))
    examples = list((root / "gallery").glob("plot_*.html"))
    generated = root / "_static" / "generated"
    if not examples or not gallery_images:
        errors.append("no rendered examples or gallery figures")
    for image in gallery_images:
        for ext in (".svg", ".pdf", ".png"):
            output = generated / image.with_suffix(ext).name
            if not output.is_file() or not output.stat().st_size:
                errors.append(f"missing figure export: {output.relative_to(root)}")
    if errors:
        raise SystemExit("\n".join(sorted(set(errors))))
    print(
        f"Checked {len(pages)} pages, {nlinks} local references, "
        f"{len(examples)} examples, and {len(gallery_images)} figures in three formats."
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
