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
    CACHE_EXPIRY_HOURS,
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
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            if age_hours > CACHE_EXPIRY_HOURS:
                logger.debug(f"Cache expired ({age_hours:.1f}h): {url}")
                path.unlink()
                return None
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

    def _cookie_path(self, login_url: str) -> Path:
        """Cookie保存パス"""
        domain = login_url.split("//")[-1].split("/")[0].replace(".", "_")
        return self.cache_dir / f"cookies_{domain}.json"

    def _load_cookies(self, login_url: str) -> bool:
        """保存済みCookieを読み込み"""
        import json
        cookie_path = self._cookie_path(login_url)
        if cookie_path.exists():
            try:
                with open(cookie_path) as f:
                    cookies = json.load(f)
                for name, value in cookies.items():
                    self.session.cookies.set(name, value)
                self._logged_in = True
                logger.info(f"Loaded saved cookies for {login_url}")
                return True
            except Exception:
                pass
        return False

    def _save_cookies(self, login_url: str) -> None:
        """Cookieをファイルに保存"""
        import json
        cookie_path = self._cookie_path(login_url)
        cookies = {c.name: c.value for c in self.session.cookies}
        with open(cookie_path, "w") as f:
            json.dump(cookies, f)

    def login(self, login_url: str, payload: dict) -> bool:
        """サイトにログインしてセッションCookieを取得（Cookie永続化対応）"""
        if self._logged_in:
            return True

        # 保存済みCookieを試す
        if self._load_cookies(login_url):
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
                self._save_cookies(login_url)
                logger.info(f"Logged in to {login_url} (cookies saved)")
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
