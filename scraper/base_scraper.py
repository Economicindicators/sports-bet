"""ベーススクレイパー: レート制限、キャッシュ、リトライ、UA回転 (keiba-aiベース)"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import urllib3
import requests
from bs4 import BeautifulSoup

# handenomori.comのSSL証明書が期限切れのため警告を抑制
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config.constants import (
    MAX_RETRIES,
    REQUEST_TIMEOUT,
    SCRAPE_DELAY_MAX,
    SCRAPE_DELAY_MIN,
    USER_AGENTS,
)
from config.settings import CACHE_DIR

logger = logging.getLogger(__name__)


class BaseScraper:
    """レート制限、キャッシュ、リトライ機能付きベーススクレイパー"""

    # SSL検証を無効にするドメイン (証明書期限切れ対応)
    SSL_VERIFY_SKIP_DOMAINS: set[str] = {"handenomori.com"}

    def __init__(self, use_cache: bool = True, cache_dir: Path | None = None):
        self.session = requests.Session()
        self.use_cache = use_cache
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0
        self._logged_in = False

    def _get_headers(self) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        key = self._cache_key(url)
        return self.cache_dir / f"{key}.html"

    def _read_cache(self, url: str) -> str | None:
        if not self.use_cache or self._logged_in:
            return None
        path = self._get_cache_path(url)
        if path.exists():
            logger.debug(f"Cache hit: {url}")
            return path.read_text(encoding="utf-8")
        return None

    def _write_cache(self, url: str, content: str) -> None:
        if not self.use_cache:
            return
        path = self._get_cache_path(url)
        path.write_text(content, encoding="utf-8")

    def _should_skip_ssl(self, url: str) -> bool:
        """SSL検証をスキップすべきドメインか判定"""
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        return any(d in host for d in self.SSL_VERIFY_SKIP_DOMAINS)

    def login(self, login_url: str, payload: dict) -> bool:
        """サイトにログインしてセッションCookieを取得"""
        if self._logged_in:
            return True
        try:
            self._rate_limit()
            r = self.session.post(
                login_url, data=payload,
                headers=self._get_headers(),
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
                verify=not self._should_skip_ssl(login_url),
            )
            self._last_request_time = time.time()
            self._logged_in = r.status_code == 200
            if self._logged_in:
                logger.info(f"Logged in to {login_url}")
            else:
                logger.warning(f"Login failed ({r.status_code})")
            return self._logged_in
        except Exception as e:
            logger.warning(f"Login error: {e}")
            return False

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        delay = random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def fetch(self, url: str) -> str:
        """URLからHTMLを取得（キャッシュ・リトライ付き）"""
        cached = self._read_cache(url)
        if cached is not None:
            return cached

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._rate_limit()
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=REQUEST_TIMEOUT,
                    verify=not self._should_skip_ssl(url),
                )
                self._last_request_time = time.time()
                response.raise_for_status()

                response.encoding = response.apparent_encoding
                content = response.text

                self._write_cache(url, content)
                return content

            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    # 404はリトライせず即スキップ (試合がない日)
                    raise
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt + random.random())
                else:
                    raise
            except requests.RequestException as e:
                logger.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {e}"
                )
                if attempt < MAX_RETRIES:
                    time.sleep(2**attempt + random.random())
                else:
                    raise

    def parse(self, html: str) -> BeautifulSoup:
        """HTMLをBeautifulSoupオブジェクトに変換"""
        return BeautifulSoup(html, "lxml")

    def fetch_and_parse(self, url: str) -> BeautifulSoup:
        """URLからHTMLを取得しパース"""
        html = self.fetch(url)
        return self.parse(html)
