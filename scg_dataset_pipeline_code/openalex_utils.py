import json
import logging
import os
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests


logger = logging.getLogger(__name__)


def load_openalex_api_key() -> Optional[str]:
    env_key = os.getenv("OPENALEX_API_KEY")
    if env_key:
        return env_key.strip()

    key_file = os.getenv("OPENALEX_API_KEY_FILE")
    candidate_paths = []
    if key_file:
        candidate_paths.append(Path(key_file).expanduser())
    candidate_paths.append(Path.home() / ".openalex_api_key")

    for path in candidate_paths:
        try:
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read OpenAlex API key file %s: %s", path, exc)
    return None


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: str, b: str) -> float:
    left = normalize_title(a)
    right = normalize_title(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def short_openalex_id(openalex_id: str) -> str:
    return str(openalex_id).rstrip("/").split("/")[-1]


def load_citation_cache(cache_file: Optional[str]) -> Dict[str, dict]:
    if not cache_file or not Path(cache_file).exists():
        return {}
    with open(cache_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): v for k, v in raw.items()}


def save_citation_cache(cache: Dict[str, dict], cache_file: Optional[str]) -> None:
    if not cache_file:
        return
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def cache_record_to_counts(record, paper_year: Optional[int], min_year: Optional[int], max_year: Optional[int]) -> Optional[Dict[int, int]]:
    """Return trimmed counts from a cache record, or None if the record should be re-queried."""
    if record is None:
        return None

    # Legacy cache format: {year: count}. Empty legacy dict is ambiguous and should be re-queried.
    if isinstance(record, dict) and "status" not in record:
        if not record:
            return None
        raw_counts = record
    elif isinstance(record, dict):
        status = record.get("status")
        if status == "api_error":
            return None
        if status not in {"matched", "not_found", "low_similarity"}:
            return None
        raw_counts = record.get("citation_counts_by_year", {}) or {}
    else:
        return None

    out = {}
    for y, c in raw_counts.items():
        y = int(y)
        if paper_year is not None and y < int(paper_year):
            continue
        if min_year is not None and y < int(min_year):
            continue
        if max_year is not None and y > int(max_year):
            continue
        out[y] = int(c)
    return out


class OpenAlexAPIError(RuntimeError):
    pass


class OpenAlexQuotaExceeded(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "y", "on"}


class OpenAlexRequestBudget:
    """Local daily request budget guard for OpenAlex paid/free API usage."""

    def __init__(
        self,
        usage_file: Optional[str] = None,
        daily_search_budget: Optional[int] = None,
        daily_list_budget: Optional[int] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and not _truthy_env("OPENALEX_DISABLE_DAILY_BUDGET")
        self.daily_search_budget = daily_search_budget or _env_int("OPENALEX_DAILY_SEARCH_BUDGET", 1000)
        self.daily_list_budget = daily_list_budget or _env_int("OPENALEX_DAILY_LIST_BUDGET", 10000)
        self.usage_file = Path(
            usage_file or os.getenv("OPENALEX_USAGE_FILE") or ".openalex_usage.json"
        ).expanduser()

    @staticmethod
    def current_date_utc() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load(self) -> dict:
        today = self.current_date_utc()
        if self.usage_file.exists():
            try:
                with open(self.usage_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date_utc") == today:
                    data.setdefault("counts", {})
                    return data
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read OpenAlex usage file %s: %s", self.usage_file, exc)
        return {
            "date_utc": today,
            "counts": {
                "search": 0,
                "list_filter": 0,
                "single_work": 0,
            },
        }

    def _save(self, data: dict) -> None:
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.usage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def reserve(self, request_kind: str) -> None:
        if not self.enabled:
            return
        if request_kind == "single_work":
            data = self._load()
            data["counts"]["single_work"] = int(data["counts"].get("single_work", 0)) + 1
            self._save(data)
            return

        budget = self.daily_search_budget if request_kind == "search" else self.daily_list_budget
        data = self._load()
        used = int(data["counts"].get(request_kind, 0))
        if used >= budget:
            raise OpenAlexQuotaExceeded(
                f"OpenAlex daily {request_kind} request budget exhausted "
                f"({used}/{budget}) for {data['date_utc']} UTC. "
                "Re-run after the daily quota resets or raise the OPENALEX_DAILY_*_BUDGET env var."
            )
        data["counts"][request_kind] = used + 1
        self._save(data)


class OpenAlexClient:
    BASE_URL = "https://api.openalex.org"

    def __init__(
        self,
        rate_limit: Optional[float] = None,
        mailto: Optional[str] = None,
        api_key: Optional[str] = None,
        title_similarity_threshold: float = 0.92,
        max_retries: int = 6,
        usage_file: Optional[str] = None,
    ):
        self.rate_limit = _env_float("OPENALEX_RATE_LIMIT_SECONDS", 0.12) if rate_limit is None else rate_limit
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO")
        self.api_key = (api_key or load_openalex_api_key() or "").strip()
        self.title_similarity_threshold = title_similarity_threshold
        self.max_retries = max_retries
        self.last_request_time = 0.0
        self.session = requests.Session()
        self.request_budget = OpenAlexRequestBudget(usage_file=usage_file, enabled=bool(self.api_key))
        if not self.api_key:
            logger.warning("OPENALEX_API_KEY is not set; OpenAlex requests will be sent without an api_key parameter.")

    def _wait_for_rate_limit(self) -> None:
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    @staticmethod
    def _request_kind(path: str, params: dict) -> str:
        if re.fullmatch(r"/works/[^/]+", path or ""):
            return "single_work"
        if "search" in params:
            return "search"
        return "list_filter"

    def _get_json(self, path: str, params: dict) -> dict:
        params = dict(params or {})
        if self.mailto:
            params["mailto"] = self.mailto
        if self.api_key:
            params["api_key"] = self.api_key
        request_kind = self._request_kind(path, params)
        last_error = None
        for attempt in range(self.max_retries):
            self.request_budget.reserve(request_kind)
            self._wait_for_rate_limit()
            try:
                response = self.session.get(f"{self.BASE_URL}{path}", params=params, timeout=45)
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else min(120 * (attempt + 1), 600)
                    logger.warning("OpenAlex rate limited (429). Waiting %ss before retry.", wait)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(5 * (2 ** attempt), 120))
                    continue
        raise OpenAlexAPIError(str(last_error))

    def find_work_by_title(self, title: str, year: Optional[int] = None) -> dict:
        params = {
            "search": title,
            "per_page": 5,
            "select": "id,display_name,publication_year,cited_by_count",
        }
        if year:
            params["filter"] = f"publication_year:{int(year)}"

        data = self._get_json("/works", params)
        results = data.get("results", []) or []
        if not results:
            return {
                "status": "not_found",
                "query_title": title,
                "publication_year": int(year) if year else None,
                "citation_counts_by_year": {},
            }

        best = max(results, key=lambda item: title_similarity(title, item.get("display_name", "")))
        sim = title_similarity(title, best.get("display_name", ""))
        if sim < self.title_similarity_threshold:
            return {
                "status": "low_similarity",
                "query_title": title,
                "publication_year": int(year) if year else None,
                "matched_title": best.get("display_name"),
                "matched_year": best.get("publication_year"),
                "title_similarity": sim,
                "citation_counts_by_year": {},
            }

        return {
            "status": "matched",
            "query_title": title,
            "publication_year": int(year) if year else None,
            "openalex_id": best.get("id"),
            "matched_title": best.get("display_name"),
            "matched_year": best.get("publication_year"),
            "title_similarity": sim,
            "cited_by_count": int(best.get("cited_by_count") or 0),
            "citation_counts_by_year": {},
        }

    def get_citation_counts_for_work(self, openalex_id: str) -> Dict[int, int]:
        work_id = short_openalex_id(openalex_id)
        params = {
            "filter": f"cites:{work_id}",
            "group_by": "publication_year",
            "per_page": 200,
        }
        data = self._get_json("/works", params)
        counts = {}
        for item in data.get("group_by", []) or []:
            key = item.get("key")
            if key is None:
                continue
            counts[int(key)] = int(item.get("count") or 0)
        return counts

    def lookup_citations_by_title(self, title: str, year: Optional[int] = None) -> dict:
        record = self.find_work_by_title(title, year)
        if record.get("status") != "matched":
            return record
        record["citation_counts_by_year"] = {
            str(y): int(c)
            for y, c in self.get_citation_counts_for_work(record["openalex_id"]).items()
        }
        return record
