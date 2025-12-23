import re
import posixpath
from urllib.parse import urlparse, urlunparse


class Api():
    def start(self):
        raise NotImplementedError
    
    def test(self, target):
        raise NotImplementedError

    def fix_target(self, url: str, default_scheme: str = "https") -> str:
        if not url.strip():
            raise ValueError("URL cannot be empty")

        # Add default scheme if missing
        if "://" not in url:
            url = f"{default_scheme}://{url}"

        parsed = urlparse(url)

        # Lowercase the scheme and hostname
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Handle user accidentally placing hostname in path (e.g. "https://example.com/foo")
        if not netloc and parsed.path:
            netloc = parsed.path
            path = ""
        else:
            path = posixpath.normpath(parsed.path)
            if path == ".":
                path = ""

        # Remove default ports
        if (scheme == "http" and netloc.endswith(":80")):
            netloc = netloc[:-3]
        elif (scheme == "https" and netloc.endswith(":443")):
            netloc = netloc[:-4]

        # Normalize path to have a trailing slash only for bare domains
        if not path:
            path = "/"
        elif not path.startswith("/"):
            path = "/" + path

        normalized = urlunparse((scheme, netloc, path, "", parsed.query, parsed.fragment))
        return normalized
